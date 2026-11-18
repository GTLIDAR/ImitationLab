"""CPU checks for resumable, bounded BONES-SEED conversion."""

import tarfile

import numpy as np
import pytest

from imitationlab.data_tools.bones_seed.csv import BONES_G1_EXPECTED_COLUMNS
from imitationlab.data_tools.bones_seed.prepare import _plan_buckets, stage_extract, stage_normalize


def test_extract_normalize_and_resume_two_clips(tmp_path):
    archive_path = tmp_path / "g1.tar.gz"
    motions = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in ("wave_left", "wave_right"):
            clip = np.zeros((2, len(BONES_G1_EXPECTED_COLUMNS)))
            clip[:, 1:4] = (25, -50, 100)
            clip[:, 6] = 90
            clip[:, 7] = 180
            source = tmp_path / f"{name}.csv"
            np.savetxt(
                source,
                clip,
                delimiter=",",
                comments="",
                header=",".join(BONES_G1_EXPECTED_COLUMNS),
            )
            member = f"g1/csv/{name}.csv"
            archive.add(source, arcname=member)
            motions.append({"filename": name, "g1_path": member})

    selection = {"motions": motions}
    raw_dir, csv_dir = tmp_path / "raw", tmp_path / "normalized"
    assert stage_extract(selection, archive_path, raw_dir)["extracted"] == 2
    assert stage_normalize(selection, raw_dir, csv_dir, workers=2)["normalized"] == 2
    for motion in motions:
        data = np.loadtxt(csv_dir / f"{motion['filename']}.csv", delimiter=",")
        np.testing.assert_allclose(data[0, :3], [0.25, -0.5, 1])
        np.testing.assert_allclose(data[0, 3:7], [0, 0, 2**-0.5, 2**-0.5])
        assert data[0, 7] == pytest.approx(np.pi)
    assert stage_extract(selection, archive_path, raw_dir)["extracted"] == 0
    assert stage_normalize(selection, raw_dir, csv_dir, workers=2)["normalized"] == 0


def test_invalid_csv_has_no_completed_output(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "bad.csv").write_text("bad header\n1\n2\n")
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="Normalization failed"):
        stage_normalize({"motions": [{"filename": "bad"}]}, raw_dir, out, workers=1)
    assert not (out / "bad.csv").exists()


def test_buckets_respect_padding_and_single_motion_limit():
    assert _plan_buckets(
        [("a", 10), ("b", 20), ("c", 30), ("d", 80)], max_envs=2, cell_budget=100
    ) == [["a", "b"], ["c"], ["d"]]
    with pytest.raises(ValueError, match="alone needs 101"):
        _plan_buckets([("oversize", 101)], max_envs=2, cell_budget=100)
