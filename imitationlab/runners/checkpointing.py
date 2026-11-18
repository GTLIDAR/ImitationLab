"""Network, optimizer, and environment checkpoint persistence for TorchRL learners."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Any

import torch


class Checkpointing:
    """Save and restore standard learner components and cumulative training state."""

    def save_model(
        self, path: str | Path | None = None, step: int | None = None
    ) -> None:
        """Save model state and training components to disk.

        Persists state dictionaries for all networks, optimizer, and environment
        statistics (if using normalization). Supports both single-file checkpoints
        and versioned checkpoints with step numbers.

        Args:
            path: Target file path or directory. If None, saves to log directory.
                If a directory, creates "model.pt" or "model_step_{step}.pt" inside.
                If a file path with suffix, uses that name (with optional step suffix).
            step: Optional training step number to append to filename for versioning.
                Creates separate checkpoint files for different training stages.

        Note:
            Saves policy, value/Q networks, feature extractor (if used), optimizer state,
            and VecNorm statistics (if environment uses observation normalization).

        Example:
            >>> algorithm.save_model("checkpoints/ppo_model.pt")
            >>> algorithm.save_model("checkpoints", step=10000)  # -> model_step_10000.pt
        """
        default_dir = self.log_dir
        target_base = Path(path).expanduser() if path else default_dir
        base_exists = target_base.exists()
        base_is_file = base_exists and target_base.is_file()
        has_suffix = bool(target_base.suffix)

        if step:
            if base_is_file or (has_suffix and not target_base.is_dir()):
                suffix = target_base.suffix
                stemmed = target_base.with_suffix("")
                target_path = stemmed.with_name(stemmed.name + f"_step_{step}{suffix}")
            else:
                target_path = target_base / f"model_step_{step}.pt"
        elif base_is_file or (has_suffix and not target_base.is_dir()):
            target_path = target_base
        else:
            target_path = target_base / "model.pt"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save: dict[str, torch.Tensor | dict] = {
            "policy_state_dict": (self.policy.state_dict() if self.policy else {}),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        if self.value_function:
            data_to_save["value_state_dict"] = self.value_function.state_dict()
        if self.q_function:
            data_to_save["q_state_dict"] = self.q_function.state_dict()
        if self.feature_extractor:
            data_to_save["feature_extractor_state_dict"] = (
                self.feature_extractor.state_dict()
            )
        # if we are using VecNorm, we need to save the running mean and std
        if (
            hasattr(self.env, "is_closed")
            and not self.env.is_closed
            and hasattr(self.env, "normalize_obs")
        ):
            data_to_save["vec_norm_msg"] = self.env.state_dict()
        if self.lr_scheduler is not None:
            data_to_save["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        data_to_save.update(self._extra_model_state_dict())
        # Global frame count for walltime-segmented resume. `step` is the loop's
        # frames_processed, which init_metadata seeds with the restored offset,
        # so this is cumulative across segments.
        if step is not None:
            data_to_save["cumulative_env_frames"] = int(step)
        data_to_save["cumulative_train_seconds"] = self.train_elapsed_seconds()

        if run_metadata := getattr(self, "_run_metadata", None):
            data_to_save["imitationlab"] = run_metadata

        torch.save(data_to_save, target_path)

    def load_model(self, path: str) -> None:
        """Load model state and training components from a saved checkpoint.

        Restores network parameters, optimizer state, and environment statistics
        from a previously saved checkpoint file. Allows resuming training or
        deploying trained models.

        Args:
            path: Path to the checkpoint file (.pt) to load.

        Note:
            Only loads components that exist in both the checkpoint and current model.
            Missing keys are safely skipped. Loads to the device specified in config.

        Example:
            >>> algorithm = PPOAlgorithm(env, config)
            >>> algorithm.load_model("checkpoints/model_step_10000.pt")
            >>> algorithm.train()  # Resume training
        """
        data = torch.load(path, map_location=self.device)
        # Continue the global frame budget on resume. Checkpoints without the
        # key have no trustworthy global count; their filename step can be
        # segment-local, so it is deliberately NOT used as a fallback.
        resume_frames = data.get("cumulative_env_frames")
        if resume_frames is not None:
            self._resume_frame_offset = int(resume_frames)
        if self.policy and "policy_state_dict" in data:
            self.policy.load_state_dict(data["policy_state_dict"])  # type: ignore[arg-type]
        if self.value_function and "value_state_dict" in data:
            self.value_function.load_state_dict(data["value_state_dict"])  # type: ignore[arg-type]
        if self.q_function and "q_state_dict" in data:
            self.q_function.load_state_dict(data["q_state_dict"])  # type: ignore[arg-type]
        optimizer_restored = "optimizer_state_dict" in data and (
            self._load_optimizer_state_dict(data["optimizer_state_dict"])
        )
        # A scheduler's base_lrs are per group, so it only fits restored groups.
        if (
            optimizer_restored
            and self.lr_scheduler is not None
            and "lr_scheduler_state_dict" in data
        ):
            self.lr_scheduler.load_state_dict(data["lr_scheduler_state_dict"])
        if self.config.feature_extractor and "feature_extractor_state_dict" in data:
            self.feature_extractor.load_state_dict(data["feature_extractor_state_dict"])  # type: ignore[arg-type]
        if hasattr(self.env, "normalize_obs") and "vec_norm_msg" in data:
            self.env.load_state_dict(data["vec_norm_msg"])  # type: ignore[arg-type]
        self._restore_train_clock(data)
        self._load_extra_model_state_dict(data)

    def _load_optimizer_state_dict(self, state_dict: Any) -> bool:
        """Restore optimizer state, tolerating a param-group-count change.

        `torch.optim.Optimizer.load_state_dict` requires the saved and live
        param groups to match one for one. Any change to how the actor's
        parameters are split into groups therefore makes every earlier
        checkpoint unloadable -- for evaluation as well as for resume, even
        though evaluation needs no optimizer state at all and the policy is
        already restored by the time this runs.

        A mismatch is reported and skipped rather than raised. The optimizer
        then starts from its fresh state: for evaluation that is exactly
        right, and for a resume it means losing the moment estimates, which
        the log says out loud because it changes the first updates after the
        restore. Returns whether the state was restored.
        """
        groups = (
            state_dict.get("param_groups") if isinstance(state_dict, dict) else None
        )
        saved_groups = len(groups) if groups is not None else None
        live_groups = len(self.optim.param_groups)
        if saved_groups is not None and saved_groups != live_groups:
            self.log.warning(
                "Checkpoint optimizer has %d param group(s) but this optimizer "
                "has %d; skipping the optimizer restore. Evaluation is "
                "unaffected (the policy is already loaded); a training resume "
                "continues with fresh optimizer moments.",
                saved_groups,
                live_groups,
            )
            return False
        self.optim.load_state_dict(state_dict)
        return True

    def _extra_model_state_dict(self) -> dict[str, Any]:
        """Algorithm-specific checkpoint payload, merged into ``save_model``."""
        return {}

    def _load_extra_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        del checkpoint

    def _should_save_checkpoint(
        self,
        *,
        frames_processed: int,
        frames_in_iteration: int,
    ) -> bool:
        """Save at frame boundaries or configured cumulative time milestones."""
        schedule = tuple(
            getattr(getattr(self.config, "trainer", None), "checkpoint_seconds", ())
        )
        due = False
        if schedule:
            if any(
                not math.isfinite(float(t)) or float(t) <= 0 for t in schedule
            ) or any(float(a) >= float(b) for a, b in zip(schedule, schedule[1:])):
                raise ValueError(
                    "checkpoint_seconds must be finite, positive, and strictly increasing"
                )
            previous = getattr(
                self,
                "_checkpoint_time_seen",
                getattr(self, "_train_seconds_offset", 0.0),
            )
            elapsed = self.train_elapsed_seconds()
            due = any(previous < float(t) <= elapsed for t in schedule)
            self._checkpoint_time_seen = elapsed
        save_interval = int(self.config.save_interval)
        if save_interval <= 0:
            return due
        current_frames = int(frames_processed)
        previous_frames = current_frames - int(frames_in_iteration)
        return due or (
            current_frames >= save_interval
            and current_frames // save_interval > previous_frames // save_interval
        )
