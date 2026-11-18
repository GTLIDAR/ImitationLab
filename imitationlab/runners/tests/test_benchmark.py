"""The benchmark must not confuse a smoke pass with measured learning."""

import json
from pathlib import Path

import pytest
import torch

from imitationlab.runners.benchmark import first_reached, motion_board, tracking_point


@pytest.mark.parametrize("policy", ["supermimic", "svg", "diffsr"])
def test_learning_curve_refuses_world_model_only_warmup(tmp_path, policy):
    from imitationlab.runners.benchmark import validate_training_checkpoint

    path = tmp_path / "model.pt"
    updated = {"state": {0: {"step": torch.tensor(10.0)}}}
    payload = {
        "policy_state_dict": {"weight": torch.ones(2)},
        "optimizers": {"wm": updated, "actor": {"state": {}}, "critic": {"state": {}}},
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Missing optimizer updates"):
        validate_training_checkpoint(path, policy)
    payload["optimizers"] = {"wm": updated}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Missing optimizer update evidence"):
        validate_training_checkpoint(path, policy)
    payload["optimizers"] = {key: updated for key in ("wm", "actor", "critic")}
    torch.save(payload, path)
    assert validate_training_checkpoint(path, policy)["optimizer_steps"]["actor"] == 10


def test_learning_curve_requires_ppo_updates(tmp_path):
    from imitationlab.runners.benchmark import validate_training_checkpoint

    path = tmp_path / "model.pt"
    torch.save({"policy_state_dict": {"weight": torch.ones(2)}}, path)
    with pytest.raises(ValueError, match="Missing optimizer updates"):
        validate_training_checkpoint(path, "ppo")
    torch.save(
        {"optimizer_state_dict": {"state": {0: {"step": torch.tensor(2.0)}}}}, path
    )
    assert validate_training_checkpoint(path, "ppo")["optimizer_steps"]["ppo"] == 2


@pytest.mark.parametrize(
    "counts", [{}, {"wm": 10}, {"wm": 10, "actor": 0, "critic": 0}]
)
def test_report_refuses_completed_cases_without_policy_updates(tmp_path, counts):
    from imitationlab.evaluation.learning_curves import summarize

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            dict(
                dataset="lafan1",
                interface="fsq",
                policy="supermimic",
                status="completed",
                checkpoint_validation=dict(optimizer_steps=counts),
            )
        )
    )
    with pytest.raises(ValueError, match="lacks positive policy optimizer"):
        summarize([path])


def bank(tmp_path, count=300):
    (tmp_path / "reference_arrays_manifest.json").write_text(
        json.dumps(
            {
                "key": {"source": {"persist_id": "test@abc"}},
                "traj_info": {
                    "start_index": list(range(count)),
                    "written": count * 500,
                },
            }
        )
    )
    return tmp_path


def test_board_is_reproducible_and_keeps_small_banks(tmp_path):
    path = bank(tmp_path)
    a = motion_board(path)
    assert a == motion_board(path)
    assert len(a["ranks"]) == 256
    assert a["ranks"] == sorted(set(a["ranks"]))
    assert a["sha256"] != motion_board(path, seed=43)["sha256"]
    bank(tmp_path, count=40)
    assert motion_board(path)["ranks"] == list(range(40))


def test_time_to_quality_is_first_observed_sample_without_interpolation():
    points = [
        dict(train_seconds=t, end_to_end_seconds=t + 100, success=s, mpjpe_l_mm=e)
        for t, s, e in [(61, 0.4, 30), (125, 0.8, 50), (306, 0.91, 42), (601, 0.96, 38)]
    ]
    assert first_reached(points, 0.5) == dict(
        seconds=125, previous_sample_seconds=61, end_to_end_seconds=225
    )
    assert first_reached(points, 0.9, local_error=40)["seconds"] == 601
    assert first_reached(points, 0.99) is None


@pytest.mark.parametrize("successes,target", [(36, 0.9), (38, 0.95)])
def test_time_to_quality_accepts_float32_boundary_without_rounding_up_failure(
    successes, target
):
    points = [
        dict(
            train_seconds=t,
            end_to_end_seconds=t + 100,
            success=torch.tensor(count / 40, dtype=torch.float32).item(),
            mpjpe_l_mm=35,
        )
        for t, count in ((600, successes - 1), (900, successes))
    ]
    assert first_reached(points[:1], target) is None
    assert first_reached(points, target, local_error=40) == dict(
        seconds=900, previous_sample_seconds=600, end_to_end_seconds=1000
    )
    points[-1]["mpjpe_l_mm"] = 40.01
    assert first_reached(points, target, local_error=40) is None


def test_report_recomputes_legacy_thresholds_without_mutating_measurements(tmp_path):
    from imitationlab.evaluation.learning_curves import summarize

    point = dict(
        train_seconds=600.5,
        end_to_end_seconds=700.5,
        frames=1000,
        success=torch.tensor(0.9).item(),
        mpjpe_l_mm=35,
        mpjpe_g_mm=90,
        survival_seconds=9.8,
        valid_transitions=19000,
        checkpoint_sha256="weights",
        evaluation_seconds=30,
    )
    recorded = {"0.5": None, "0.9": None, "0.95": None, "0.9_and_40mm": None}
    raw = dict(
        dataset="lafan1",
        interface="explicit",
        policy="ppo",
        status="completed",
        board=motion_board(bank(tmp_path, 40)),
        source_sha256="training-source",
        budget_seconds=600,
        milestones=[600],
        seed=42,
        points=[point],
        pretrain_seconds=0,
        startup_seconds=100,
        thresholds=recorded,
        checkpoint_validation=dict(optimizer_steps={"ppo": 10}),
    )
    path = tmp_path / "result.json"
    path.write_text(json.dumps(raw))
    original = path.read_bytes()
    case = summarize([path])["cases"][0]
    assert case["thresholds"]["0.9"]["seconds"] == 600.5
    assert case["thresholds"]["0.9_and_40mm"]["seconds"] == 600.5
    assert case["thresholds"]["0.95"] is None
    assert case["recorded_thresholds"] == recorded
    assert case["points"] == [point]
    assert path.read_bytes() == original


def evaluation(board):
    return {
        "metadata": dict(
            task="imitation-g1-v1",
            steps_requested=board["steps"],
            seed=42,
            reference_start_frame=0,
            action_sampling="mode",
            randomization_profile="none",
            early_terminations_enabled=True,
            keep_after_done=False,
            reference_arrays=dict(persist_id=board["bank_id"]),
        ),
        "aggregate": dict(
            num_evaluated_envs=2,
            valid_transition_count=200,
            tracking_success_rate=0.5,
            survival_steps_mean=50,
            steps_executed=100,
        ),
        "metrics": dict(
            tracking_mpjpe_mm=dict(mean=50), tracking_mpjpe_g_mm=dict(mean=100)
        ),
        "per_environment": [dict(trajectory_rank=rank) for rank in board["ranks"]],
    }


def test_tracking_requires_the_same_population_and_protocol(tmp_path):
    board = motion_board(bank(tmp_path, 2), steps=100)
    raw = evaluation(board)
    point = tracking_point(raw, dict(train_seconds=60.4), board)
    assert point["success"] == 0.5 and point["survival_seconds"] == 1
    raw["metadata"]["steps_requested"] = 500
    with pytest.raises(ValueError, match="protocol"):
        tracking_point(raw, {}, board)
    raw = evaluation(board)
    raw["per_environment"][0]["trajectory_rank"] = 999
    with pytest.raises(ValueError, match="ranks"):
        tracking_point(raw, {}, board)
    raw = evaluation(board)
    raw["metrics"]["tracking_mpjpe_mm"]["mean"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        tracking_point(raw, {}, board)


def test_report_uses_actual_times_and_preserves_missing_results():
    from imitationlab.evaluation.learning_curves import render_markdown, sample_at

    points = [dict(train_seconds=121.5, success=0.5, mpjpe_l_mm=32)]
    assert sample_at(points, 60) is sample_at(points, 120)
    assert sample_at(points, 300) is None
    text = render_markdown(dict(cases=[], sources=[], gpus=[], boards={}))
    assert "0/24 cases completed" in text
    assert text.count("| pending | pending | pending | pending | pending |") == 24
    assert "not reached" not in text.split("## Protocol")[0]


def test_report_includes_thirty_minute_checkpoints():
    from imitationlab.evaluation.learning_curves import render_markdown

    point = dict(
        train_seconds=1800.8,
        success=0.95,
        mpjpe_l_mm=30,
        mpjpe_g_mm=80,
        survival_seconds=10,
        end_to_end_seconds=2000.8,
    )
    case = dict(
        dataset="lafan1",
        interface="explicit",
        policy="ppo",
        status="completed",
        budget_seconds=1800,
        seed=42,
        milestones=[60, 120, 300, 600, 900, 1200, 1800],
        points=[point],
        thresholds={},
        profile="standard",
        resolved={},
        pretrain_seconds=0,
        startup_seconds=200,
    )
    text = render_markdown(dict(cases=[case], sources=[], gpus=[], boards={}))
    assert "≥10 min | ≥15 min | ≥20 min | ≥30 min" in text
    assert "95.0% / 30.0 @ 30.01" in text


def test_benchmark_defaults_to_thirty_training_minutes(monkeypatch, tmp_path):
    from imitationlab.runners import benchmark

    def run(args, rest):
        assert args.seconds == 1800
        assert args.milestones == [60, 120, 300, 600, 900, 1200, 1800]
        assert rest == []
        return 0

    monkeypatch.setattr(benchmark, "run", run)
    assert (
        benchmark.main(
            [
                "run",
                "--dataset",
                "lafan1",
                "--data",
                str(tmp_path),
                "--policy",
                "ppo",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "mismatch", ["seed", "source_sha256", "interface", "bank_sha256", "weights"]
)
def test_frozen_encoder_reuse_requires_pinned_identity(tmp_path, mismatch):
    from imitationlab.runners.artifacts import file_sha256
    from imitationlab.runners.benchmark import frozen_encoder_record

    weights = tmp_path / "encoder.pt"
    weights.write_bytes(b"frozen encoder")
    record = dict(
        status="completed",
        interface="fsq",
        seed=42,
        source_sha256="a" * 64,
        board=dict(bank_sha256="bank"),
        encoder=str(weights),
        encoder_sha256=file_sha256(weights),
    )
    path = tmp_path / "pretrain.json"
    path.write_text(json.dumps(record))
    expected = dict(
        interface="fsq", seed=42, source_sha256="a" * 64, board=dict(bank_sha256="bank")
    )
    assert frozen_encoder_record(tmp_path, **expected) == record
    if mismatch == "weights":
        weights.write_bytes(b"different encoder")
    elif mismatch == "bank_sha256":
        expected["board"] = dict(bank_sha256="different bank")
    else:
        expected[mismatch] = 43 if mismatch == "seed" else "different"
    with pytest.raises(ValueError, match="does not match|weights changed"):
        frozen_encoder_record(tmp_path, **expected)


@pytest.mark.parametrize("eval_seed", [None, 42])
def test_benchmark_can_hold_board_fixed_when_policy_seed_changes(
    monkeypatch, tmp_path, eval_seed
):
    from imitationlab.runners import benchmark

    data = bank(tmp_path)
    selected_seed = 43 if eval_seed is None else eval_seed
    board = motion_board(data, count=2, seed=selected_seed, steps=100)
    output = tmp_path / "run"
    observed = []

    def child(command, log, timeout):
        observed.append(command)
        seed = int(command[command.index("--seed") + 1])
        if "train" in command:
            assert seed == 43
            run_dir = output / "training" / "trial"
            (run_dir / "models").mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    dict(
                        policy="ppo",
                        interface="explicit",
                        task="imitation-g1-v1",
                        seed=seed,
                        data=dict(persist_id=board["bank_id"]),
                    )
                )
            )
            (run_dir / "training_clock.json").write_text(
                json.dumps(dict(monotonic_start=10))
            )
            torch.save(
                dict(
                    cumulative_train_seconds=1.2,
                    optimizer_state_dict=dict(state={0: dict(step=torch.tensor(2.0))}),
                ),
                run_dir / "models" / "policy_step_1000.pt",
            )
        else:
            assert seed == selected_seed
            ranks = command[command.index("--trajectory_ranks") + 1 :]
            assert list(map(int, ranks)) == board["ranks"]
            raw = evaluation(board)
            raw["metadata"]["seed"] = seed
            Path(command[command.index("--output") + 1]).write_text(json.dumps(raw))
        return dict(exit_code=0, wall_seconds=12, monotonic_start=0)

    monkeypatch.setattr(benchmark, "child", child)
    flags = [] if eval_seed is None else ["--eval-seed", str(eval_seed)]
    assert (
        benchmark.main(
            [
                "run",
                "--dataset",
                "lafan1",
                "--data",
                str(data),
                "--policy",
                "ppo",
                "--output",
                str(output),
                "--seed",
                "43",
                "--seconds",
                "1",
                "--milestones",
                "1",
                "--eval-clips",
                "2",
                "--eval-steps",
                "100",
                *flags,
            ]
        )
        == 0
    )
    result = json.loads((output / "result.json").read_text())
    assert result["seed"] == 43 and result["board"] == board
    assert len(observed) == 2


def test_overview_aggregates_cases_on_their_shared_time_span():
    from imitationlab.evaluation.learning_curves import overview_curves

    def case(policy, points):
        return dict(
            policy=policy,
            points=[dict(train_seconds=t, success=s) for t, s in points],
        )

    summary = dict(
        cases=[
            case("ppo", [(60, 0.0), (600, 0.5), (1800, 1.0)]),
            case("ppo", [(120, 0.2), (1800, 0.6)]),
            case("ppo", [(120, 0.4), (1200, 0.8)]),
            case("svg", []),
        ]
    )
    curves = overview_curves(summary, "success")
    assert set(curves) == {"ppo"}
    ppo = curves["ppo"]
    assert ppo["cases"] == 3
    assert ppo["minutes"][0] == pytest.approx(2.0)
    assert ppo["minutes"][-1] == pytest.approx(20.0)
    assert ppo["low"][-1] == pytest.approx(100 * (0.2 + 0.4 * 1080 / 1680))
    assert ppo["high"][-1] == pytest.approx(80.0)
    assert ppo["median"][-1] == pytest.approx(100 * (0.5 + 0.5 * 600 / 1200))
