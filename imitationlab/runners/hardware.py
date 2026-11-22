"""Local hardware discovery and transparent execution-size recommendations.

The budgets are conservative estimates, not device benchmarks. They never
change model dimensions, rewards, precision, horizons, or learning rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path

from imitationlab.runners.registry import LATENT_INTERFACES

GIB = 2**30
PROFILE_NAMES = ("small", "standard", "large", "xlarge")
# Same env:learner-batch ratio as the standard fast configuration.
PROFILES = {
    "small": (1024, 512, 6.0),
    "standard": (4096, 2048, 12.0),
    "large": (8192, 4096, 25.0),
    "xlarge": (16384, 8192, 48.0),
}


@dataclass(frozen=True)
class Hardware:
    cpu_count: int
    ram_total: int
    ram_available: int
    device: str = "cuda:0"
    gpu_name: str | None = None
    vram_total: int = 0
    vram_free: int = 0
    compute_capability: tuple[int, int] | None = None
    torch_version: str = ""
    cuda_version: str | None = None
    notes: tuple[str, ...] = ()


def _read_number(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def detect_hardware(device: str = "cuda:0") -> Hardware:
    import psutil

    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    limit = _read_number("/sys/fs/cgroup/memory.max")
    current = _read_number("/sys/fs/cgroup/memory.current")
    if limit is not None and current is not None and limit < total:
        total, available = limit, min(available, max(0, limit - current))
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    fields = dict(
        cpu_count=cpus, ram_total=total, ram_available=available, device=device
    )
    try:
        import torch

        fields.update(torch_version=torch.__version__, cuda_version=torch.version.cuda)
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            return Hardware(
                **fields,
                notes=(
                    "A CUDA GPU is required for automatic G1 training configuration.",
                ),
            )
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        props = torch.cuda.get_device_properties(index)
        free, gpu_total = torch.cuda.mem_get_info(index)
        # The training-only memory emulation setting must also constrain selection.
        emulated = os.environ.get("IMITATION_EMULATE_GPU_GIB")
        if emulated:
            free = min(free, int(float(emulated) * GIB))
        return Hardware(
            **fields,
            gpu_name=props.name,
            vram_total=gpu_total,
            vram_free=free,
            compute_capability=(props.major, props.minor),
        )
    except (ImportError, OSError, RuntimeError, ValueError, AssertionError) as exc:
        return Hardware(**fields, notes=(f"CUDA discovery failed: {exc}",))


@dataclass(frozen=True)
class MotionBank:
    kind: str = "unknown"
    persist_id: str | None = None
    rows: int = 0
    library_bytes: int = 0
    manifest_sha256: str | None = None


def inspect_bank(path: str | Path | None) -> MotionBank:
    if not path:
        return MotionBank()
    path = Path(path).expanduser()
    if path.is_file():
        return MotionBank(
            kind="manifest",
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    manifest = path / "reference_arrays_manifest.json"
    if not manifest.is_file():
        return MotionBank()
    content = manifest.read_bytes()
    obj = json.loads(content)
    info = obj.get("traj_info", {})
    rows = int(info.get("written", info.get("capacity", 0)))
    # G1 v1's compact reference library is 71 float32 values per bank frame.
    return MotionBank(
        kind="packed",
        persist_id=obj["key"]["source"]["persist_id"],
        rows=rows,
        library_bytes=rows * 71 * 4,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
    )


@dataclass
class HardwarePlan:
    hardware: Hardware
    bank: MotionBank
    profile: str | None
    stage: str
    policy: str
    interface: str
    overrides: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    runtime_budget_bytes: int = 0
    calibration: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def choose_plan(
    hardware: Hardware,
    bank: MotionBank,
    *,
    policy: str = "supermimic",
    interface: str = "explicit",
    stage: str = "train",
    profile: str = "auto",
) -> HardwarePlan:
    if profile not in ("auto", "off", *PROFILE_NAMES):
        raise ValueError(f"Unknown hardware profile: {profile}")
    plan = HardwarePlan(hardware, bank, None, stage, policy, interface)
    if profile == "off":
        plan.reasons.append(
            "Hardware selection disabled; use recipe and caller settings."
        )
        return plan
    if hardware.gpu_name is None:
        plan.errors.extend(hardware.notes or ("No CUDA GPU detected.",))
        return plan
    reserve = max(GIB, int(hardware.vram_total * 0.08))
    usable = max(0, hardware.vram_free - reserve)
    # Latent interfaces keep their existing larger uncompiled networks. DiffSR
    # dynamics also need space for the feature head and denoising activations.
    if policy == "ppo":
        # PPO retains full observation rollouts and collector/replay copies.
        # Its 4096-env explicit rollout exceeded 17 GiB on the local A4500,
        # before optimization. Latent observations reduce this footprint.
        factor = 2.0 if interface == "explicit" else 1.6
    else:
        factor = (1.25 if interface != "explicit" else 1.0) * (
            1.15 if policy == "diffsr" else 1.0
        )
    if stage == "pretrain":
        factor = 1.0
    candidates = [
        name for name in PROFILE_NAMES if PROFILES[name][2] * factor * GIB <= usable
    ]
    if stage == "train" and bank.kind == "packed" and bank.rows:
        resident = [
            name
            for name in candidates
            if PROFILES[name][2] * factor * GIB + bank.library_bytes <= usable
        ]
        # Prefer GPU residency without reducing the standard fast configuration
        # to small solely to fit the library (e.g. full BONES on a 24 GiB GPU).
        if resident and (resident[-1] != "small" or candidates[-1] == "small"):
            candidates = resident
        elif bank.library_bytes + 4 * GIB > hardware.ram_available:
            candidates = resident
    if not candidates and profile == "auto":
        plan.errors.append(
            f"No automatic profile fits available GPU/host memory with a {bank.library_bytes / GIB:.1f} GiB compact library. Free memory, use a smaller bank, or supply manual settings with --hardware off."
        )
        return plan
    name = candidates[-1] if profile == "auto" else profile
    if (
        profile == "auto"
        and stage == "train"
        and hardware.gpu_name.startswith("NVIDIA H100")
        and policy in ("svg", "diffsr")
        and interface in LATENT_INTERFACES
        and PROFILE_NAMES.index(name) > PROFILE_NAMES.index("standard")
    ):
        # Ten-minute H100 checks improve LAFAN1 learning and BONES DiffSR,
        # with a documented BONES SVG tradeoff. Use one default across banks,
        # not a claim of an optimal profile for every motion distribution.
        name = "standard"
        plan.reasons.append(
            "H100 latent SVG/DiffSR defaults cap execution at standard based on the published learning-speed checks; explicit profile overrides still take precedence."
        )
    envs, batch, budget_gib = PROFILES[name]
    plan.profile = name
    plan.runtime_budget_bytes = int(budget_gib * factor * GIB)
    if plan.runtime_budget_bytes > usable:
        plan.errors.append(
            f"Profile {name} exceeds the estimated available GPU budget. Use a smaller profile or manual settings."
        )
        return plan
    if hardware.ram_available < 4 * GIB:
        plan.errors.append(
            "Less than 4 GiB of host RAM is available; free host memory before launching."
        )
        return plan
    compile_threads = max(
        1, min(8, hardware.cpu_count, hardware.ram_available // (2 * GIB))
    )
    plan.environment["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
    plan.reasons.append(
        f"Selected {name} from {hardware.vram_free / GIB:.1f} GiB free VRAM with {reserve / GIB:.1f} GiB reserved as headroom."
    )
    if stage == "pretrain":
        batch = {"small": 32, "standard": 128, "large": 256, "xlarge": 512}[name]
        plan.overrides = [
            "--num_envs",
            str(min(4, hardware.cpu_count)),
            "--batch_size",
            str(batch),
        ]
        plan.reasons.append(
            "Encoder network dimensions and training objective stay unchanged."
        )
        return plan
    # GPU-resident construction is chunked; only the host-backed placement
    # requires the full compact bank pinned in RAM.
    if bank.kind == "packed" and bank.rows:
        gpu_fits = plan.runtime_budget_bytes + bank.library_bytes <= usable
        host_fits = bank.library_bytes + 4 * GIB <= hardware.ram_available
        if not gpu_fits and not host_fits:
            plan.errors.append(
                f"The {bank.library_bytes / GIB:.1f} GiB reference library plus host workspace exceeds available RAM ({hardware.ram_available / GIB:.1f} GiB). Close applications or use a smaller bank."
            )
            return plan
        plan.overrides.extend(
            [
                f"env.library_host_mapped={'false' if gpu_fits else 'true'}",
                f"env.library_env_window={0 if gpu_fits else 64}",
            ]
        )
        plan.reasons.append(
            f"The {bank.library_bytes / GIB:.2f} GiB compact reference library uses {'GPU memory' if gpu_fits else 'host memory with 64-frame GPU windows'}; full arrays remain CPU-mapped."
        )
    elif bank.kind == "unknown":
        plan.reasons.append(
            "Bank size is unknown; existing reference storage settings are retained."
        )
    plan.overrides.extend(["--num_envs", str(envs)])
    if policy != "ppo":
        plan.overrides.extend(
            [f"agent.wm.imagine_batch={batch}", f"agent.wm.wm_batch_size={batch}"]
        )
    plan.reasons.append(
        "These are hardware-based estimates; optional calibration measures throughput without asserting tracking quality."
    )
    return plan


def selected_device(argv: list[str]) -> str:
    from imitationlab.runners.artifacts import flag_value

    return str(flag_value(argv, "--device", "cuda:0"))


def bank_from_argv(argv: list[str]) -> MotionBank:
    values = dict(
        item.split("=", 1)
        for item in argv
        if item.startswith("env.data.") and "=" in item
    )
    for key in ("env.data.reference_arrays_dir", "env.data.manifest"):
        value = values.get(key, "").strip("'\"")
        if value.lower() not in ("", "none", "null"):
            return inspect_bank(value)
    return MotionBank()


def plan_for_argv(argv: list[str], **kwargs) -> HardwarePlan:
    from imitationlab.runners.args_file import expand_args_files

    expanded = expand_args_files(argv, environ={})
    return choose_plan(
        detect_hardware(selected_device(expanded)), bank_from_argv(expanded), **kwargs
    )


def explain(plan: HardwarePlan) -> str:
    gpu = plan.hardware.gpu_name or "no CUDA GPU"
    prefix = f"[ImitationLab] {gpu}; {plan.hardware.ram_available / GIB:.1f} GiB available RAM; {plan.hardware.cpu_count} CPUs"
    return "\n".join(
        [prefix, *[f"[ImitationLab] {s}" for s in plan.reasons + plan.errors]]
    )
