"""Exercise the complete host-side launcher without allocating a GPU."""

import json
import os
import shlex
import shutil
import subprocess

from imitationlab.paths import REPO_ROOT


def test_standalone_container_launch_preserves_assignment_and_private_caches(tmp_path):
    workspace = tmp_path / "workspace"
    script = workspace / "docker/cluster/run_singularity.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "docker/cluster/run_singularity.sh", script)
    image = tmp_path / "runtime.sif"
    image.touch()
    values = dict(
        CLUSTER_DATA_DIR=tmp_path / "data",
        CLUSTER_ISAAC_SIM_CACHE_DIR=tmp_path / "cache",
        CLUSTER_ISAACLAB_DIR=tmp_path / "persistent",
        CLUSTER_SIF_PATH=tmp_path,
        CLUSTER_SHARED_SIF_PATH=image,
        CLUSTER_JOB_TMPDIR_ROOT=tmp_path / "jobs",
        CLUSTER_USE_SHARED_SIF="1",
        CLUSTER_SKIP_CACHE_COPY="1",
        CLUSTER_AUTO_SETUP_G1_DATA="0",
        CLUSTER_PYTHON_EXECUTABLE="-m",
        CLUSTER_EXTRA_PYTHONPATH_REL=".",
        REMOVE_CODE_COPY_AFTER_JOB="true",
    )
    (script.parent / "job_env.resolved.sh").write_text(
        "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in values.items())
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    stub = binaries / "apptainer"
    result = tmp_path / "container.json"
    stub.write_text(
        '#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["TEST_RESULT"]).write_text(json.dumps(dict(argv=sys.argv[1:], gpu=os.environ.get("APPTAINERENV_CUDA_VISIBLE_DEVICES"), job=os.environ.get("APPTAINERENV_SLURM_JOB_ID"))))\n'
    )
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{binaries}:" + os.environ["PATH"],
        TEST_RESULT=str(result),
        SLURM_JOB_ID="123",
        CUDA_VISIBLE_DEVICES="7",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "private-inductor"),
        TRITON_CACHE_DIR=str(tmp_path / "private-triton"),
    )
    proc = subprocess.run(
        ["bash", str(script), str(workspace), "runtime", "imitationlab", "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    called = json.loads(result.read_text())
    assert called["gpu"] == "7" and called["job"] == "123"
    argv = called["argv"]
    assert any(x.startswith(str(tmp_path / "private-inductor") + ":") for x in argv)
    assert any(x.startswith(str(tmp_path / "private-triton") + ":") for x in argv)
    assert any(":/isaac-sim/kit/cache:rw" in x for x in argv)
    assert str(image) in argv
    assert not (workspace / "docker/.env.base").exists()
