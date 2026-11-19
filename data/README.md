# Motion data

Keep source motions and generated banks outside Git. The repository includes
one Unitree Dance102 clip and a [manifest template](../imitationlab/envs/isaaclab/manifests/g1_npz_manifest.template.json).

## LAFAN1 G1

The [Unitree retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
provides 30 Hz G1 CSVs. Conversion replays them in Isaac Sim, which requires
accepting the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Kit asks on first use in a terminal; for scripts and background jobs, set
`OMNI_KIT_ACCEPT_EULA=YES` after reading it. Download and convert the
motions to 50 Hz NPZs with:

```bash
pixi run --locked -e isaaclab imitationlab data prepare lafan1 \
  --data-root data/lafan1 --batch-size 64
```

For an existing CSV directory, add `--csv-dir /path/to/g1`. To use the
already downloaded files under `data/lafan1/raw/g1/`, add
`--skip-download`. Each Isaac batch converts motions in parallel. Rerun
with the same inputs and options to continue an interrupted conversion.
`--npz-dir` and `--manifest-path` override the default outputs,
`data/lafan1/npz/g1/` and `data/lafan1/manifests/g1_lafan1_manifest.json`.

To trim an arms-up alignment pose before conversion, add
`--auto-trim-mode g1_shoulder_roll`. Changing frame rates, trims, or inputs
requires a fresh output directory or `--overwrite-npz`.

## BONES-SEED G1

BONES-SEED is proprietary motion data from Bones Studio. Each collaborator
needs authorized access to the
[gated source dataset](https://huggingface.co/datasets/bones-studio/seed/tree/main).
Keep the archive, metadata, converted motions, and packed arrays in local or
otherwise authorized storage outside Git. Review the
[license](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md)
before sharing trained weights, demos, or other results.

After obtaining access, download the archive and metadata directly:

```bash
RAW=data/bones_seed/raw
pixi run --locked hf download bones-studio/seed \
  g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset --local-dir "$RAW"
```

Convert them with the staged pipeline. The `convert` stage runs Isaac Sim
and needs the same EULA acceptance as LAFAN1:

```bash
WORK=data/bones_seed/converted
pixi run --locked -e isaaclab imitationlab data prepare bones_seed \
  --metadata-csv "$RAW/metadata/seed_metadata_v004.csv" \
  --archive "$RAW/g1.tar.gz" --work-root "$WORK" --device cuda:0 \
  --normalize-workers 8 --anchor-workers 4 \
  --max-envs 512 --cell-budget 500000
```

The pipeline applies the SONIC motion filter, records a selection hash, and
converts 120 Hz source motions to 50 Hz by default. You can pass an existing
selection JSON with `--selection`. Extraction, normalization, conversion,
anchoring, auditing, and manifest generation can run as separate
`--stages`. Normalization and anchoring use CPU worker processes. Isaac
converts length-sorted batches on the selected local GPU.

Isaac writes temporary USD files under `$TMPDIR` (default `/tmp`) during
conversion. If `/tmp` is a small RAM-backed or quota-limited filesystem,
point `TMPDIR` at a directory on disk before running the pipeline.

Rerun with the same selection, rates, and work root to skip completed
stages or clips. Use a separate work root when testing with `--limit` or
changing source or rate settings. The training manifest is
`$WORK/manifests/g1_bones_seed_sonic_full_manifest.json`.

### Distribution

ImitationLab publishes this conversion pipeline, not converted BONES-SEED
data. Run the pipeline from your own licensed copy. The published results
used the packed bank `bones_seed_sonic_full_129785@e714bbff` (129,785
motions, 47,491,234 transitions). The bank's `reference_arrays_manifest.json`
records this identity and the per-motion layout. Its SHA-256 also covers the
absolute source paths, so a bank rebuilt in another directory has a different
manifest hash. To check that two copies hold the same motions, compare their
content fingerprints (see below), which ignore paths and persist IDs.

To share a converted bank only with collaborators covered by the same
license, `python -m imitationlab.data_tools.reference_arrays.publish push`
uploads it to a Hugging Face dataset you control, and `publish fetch`
downloads and validates it. `push` uploads privately by default, refuses to
change an existing repo's visibility, and refuses a public upload of a bank
labeled BONES-SEED.

## Validate and pack

Check that a manifest resolves and its clips have usable frame rates,
array shapes, and frame ranges:

```bash
pixi run --locked imitationlab data validate "$MANIFEST"
```

For packed training, use an NPZ manifest with modern `qpos`, `qvel`, and
named bodies and joints. Bake any trims into the NPZ files before packing.
The legacy Dance102 clip works directly through its manifest.

```bash
PERSIST_ID="g1@$(sha256sum "$MANIFEST" | cut -c1-12)"
ARRAYS="data/reference_arrays/${PERSIST_ID}"
pixi run --locked imitationlab data pack "$MANIFEST" \
  --output-dir "$ARRAYS" --persist-id "$PERSIST_ID" \
  --anchor-body pelvis --workers 4 --verify-load \
  --body-names pelvis left_hip_roll_link left_knee_link left_ankle_roll_link \
    right_hip_roll_link right_knee_link right_ankle_roll_link torso_link \
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link \
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link
```

For BONES-SEED, add `--dataset-name bones_seed_sonic_full`. Keep the persist
ID with the bank. Validate a completed bank's identity, sidecar, trajectory
spans, file sizes, and sampled values before using it:

```bash
pixi run --locked imitationlab data validate "$ARRAYS" --persist-id "$PERSIST_ID"
```

Add `--manifest "$MANIFEST" --verify-source` to check the source manifest
hash and compare sampled values against the NPZs. Add `--scan` to check all
array values for non-finite entries. Add `--fingerprint` to print a SHA-256
of the bank's layout and array bytes that does not depend on where it was
built or its persist ID, and `--expected-fingerprint` to require one. The
fingerprint matches only for byte-identical arrays; converting on different
hardware or software versions can change floating-point values. Use the validated manifest or bank with
the training commands in [the training guide](../docs/training.md).
