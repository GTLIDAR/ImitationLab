#!/usr/bin/env python3
"""Transfer a packed reference-array directory through a Hugging Face dataset.

Uploads are private by default. Only upload arrays you have permission to
distribute to the destination's audience. Keep proprietary BONES-SEED assets
and derived arrays in authorized storage outside Git and public repositories.
A push refuses an existing repo whose visibility differs from the request, and
refuses ``--public`` for a bank whose sidecar labels name BONES-SEED.

Push and fetch validate the sidecar identity and each array's size. Files
remain in the same layout used by the local memory-mapped reader.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from imitationlab.data_tools.reference_arrays.build import validation_errors
from imitationlab.data_tools.reference_arrays.schema import SIDECAR_NAME


CARD_NAME = "README.md"


def _read_sidecar(directory: Path) -> dict[str, Any]:
    try:
        return json.loads((directory / SIDECAR_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not read {directory / SIDECAR_NAME}: {error}"
        ) from error


def _gate(
    directory: Path,
    *,
    persist_id: str,
    body_names: list[str] | None,
    anchor_body: str | None,
    expected_motions: int | None,
    expected_transitions: int | None,
    what: str,
) -> dict[str, Any]:
    errors = validation_errors(
        directory,
        persist_id=persist_id,
        body_names=body_names,
        anchor_body=anchor_body,
        expected_motions=expected_motions,
        expected_transitions=expected_transitions,
    )
    if errors:
        raise RuntimeError(
            f"Refusing to {what} {directory}: "
            + "; ".join(errors)
            + ". Rebuild or re-download it rather than shipping a partial artifact."
        )
    return _read_sidecar(directory)


def dataset_card(sidecar: dict[str, Any], *, source_repo: str | None) -> str:
    """A card that says what the artifact is and how to consume it."""
    key, traj = sidecar["key"], sidecar["traj_info"]
    arrays = key["arrays"]
    total = sum(
        int(_prod(spec["shape"])) * (4 if spec["dtype"] == "float32" else 8)
        for spec in arrays.values()
    )
    rows = "\n".join(
        f"| `{name}` | {tuple(spec['shape'])} | {spec['dtype']} | "
        f"{spec.get('quaternion_order') or '-'} |"
        for name, spec in sorted(arrays.items())
    )
    source_line = (
        f"Built from [`{source_repo}`](https://huggingface.co/datasets/{source_repo})."
        if source_repo
        else "Built from the source NPZ tree named in the sidecar."
    )
    return f"""---
license: other
tags:
- robotics
- motion-capture
- humanoid
---

# Reference arrays: `{key["source"]["persist_id"]}`

Training-shaped reference arrays for humanoid motion tracking:
**{len(traj["ordered_traj_list"]):,} trajectories, {traj["written"]:,} transitions,
{total / 1e9:.1f} GB**. {source_line}

This is a *derived* artifact, not a source dataset. It holds exactly the tensors
an imitation environment consumes at run time, in their final layout, so a
consumer memory-maps them instead of rebuilding a Zarr and a replay buffer. The
authority on its identity is `{SIDECAR_NAME}`, which also carries the full
trajectory table (`start_index` / `end_index` / `ordered_traj_list`) — without
it the arrays are unloadable.

Access and redistribution are subject to the source dataset's terms.

## Contents

| array | shape | dtype | quaternion order |
| --- | --- | --- | --- |
{rows}

Retained bodies (order is the column layout, not a set):
{", ".join(f"`{n}`" for n in key["body_names"])}

Baked anchor body: `{key["anchor_body"]}`. Any other anchor **within the
retained set** is derived at load time from the body arrays, so this artifact
serves those too.

`joint_pos` and `joint_vel` are deliberately absent: they are `qpos[:, 7:]` and
`qvel[:, 6:]`.

## Provenance

- source manifest SHA-256: `{key.get("manifest_sha256", "-")}`
- joints ({len(key.get("joint_names", []))}): recorded in the sidecar
- full dataset body order ({len(key.get("dataset_body_names", []))}): recorded in the sidecar

## Use

```bash
python -m imitationlab.data_tools.reference_arrays.publish fetch \\
  --repo-id <this repo> --dest-dir /path/on/compute/node \\
  --persist-id {key["source"]["persist_id"]} \\
  --expected-motions {len(traj["ordered_traj_list"])} \\
  --expected-transitions {traj["written"]}
```

Then point the environment at it:

```text
env.data.manifest=null
env.data.reference_arrays_dir=/path/on/compute/node
env.data.runtime_cache_device=cpu
env.data.macro_cache_device=cuda:0
```

The fetch validates the sidecar and every array's size against it. It does not
re-verify contents against the source NPZs; do that at build time with
`python -m imitationlab.data_tools.reference_arrays.build --verify-load`.
"""


def _prod(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def restricted_sources(sidecar: dict[str, Any]) -> list[str]:
    """Return the sidecar labels that mark a bank as derived from BONES-SEED.

    The labels checked are each trajectory's dataset label and the persist ID.
    Both are chosen at build time, so this check catches a mislabeled bank
    only when one of them names the source.
    """
    traj = sidecar.get("traj_info", {})
    labels = {
        str(entry[0])
        for entry in traj.get("ordered_traj_list", [])
        if isinstance(entry, (list, tuple)) and entry
    }
    persist_id = sidecar.get("key", {}).get("source", {}).get("persist_id")
    if persist_id:
        labels.add(str(persist_id))
    return sorted(label for label in labels if "bones" in label.lower())


def _ensure_repo(api: Any, repo_id: str, *, private: bool) -> None:
    """Create the dataset repo, or refuse one whose visibility differs.

    ``create_repo(exist_ok=True)`` leaves an existing repo's visibility as it
    is, so a private push into an existing public repo would be public.
    """
    wanted = "private" if private else "public"
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(repo_id, repo_type="dataset", private=private)
        return
    current = api.repo_info(repo_id, repo_type="dataset").private
    if current is None or bool(current) != private:
        if current is None:
            state = "of unknown visibility"
        else:
            state = "private" if current else "public"
        raise RuntimeError(
            f"Refusing to push to {repo_id}: the repo exists and is {state}, but "
            f"the push requested a {wanted} repo. Change its visibility on the "
            "Hub or choose another repo."
        )


def push(
    *,
    source_dir: Path,
    repo_id: str,
    persist_id: str,
    body_names: list[str] | None = None,
    anchor_body: str | None = None,
    expected_motions: int | None = None,
    expected_transitions: int | None = None,
    private: bool = True,
    source_repo: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate, write a card, and upload the directory."""
    sidecar = _gate(
        source_dir,
        persist_id=persist_id,
        body_names=body_names,
        anchor_body=anchor_body,
        expected_motions=expected_motions,
        expected_transitions=expected_transitions,
        what="publish",
    )
    if not private:
        restricted = restricted_sources(sidecar)
        if restricted:
            raise RuntimeError(
                f"Refusing a public push of {source_dir}: it is derived from "
                f"BONES-SEED ({', '.join(restricted)}), whose license does not "
                "permit public redistribution. Push it privately instead."
            )
    card = dataset_card(sidecar, source_repo=source_repo)
    files = sorted(p.name for p in source_dir.iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in source_dir.iterdir() if p.is_file())
    print(
        f"[PLAN] push {source_dir} -> {repo_id} "
        f"({len(files)} files, {total / 1e9:.1f} GB, private={private})"
    )
    for name in files:
        print(f"         {name}")
    if dry_run:
        print("[DRY RUN] nothing uploaded")
        return {"card": card, "files": files, "bytes": total}

    # Keep large transfers on the existing non-Xet path by default.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, upload_large_folder

    _ensure_repo(HfApi(), repo_id, private=private)
    (source_dir / CARD_NAME).write_text(card, encoding="utf-8")
    # upload_large_folder resumes from <folder>/.cache/huggingface and can skip
    # files after the repo is recreated; verify_remote catches that case.
    upload_large_folder(
        folder_path=str(source_dir), repo_id=repo_id, repo_type="dataset"
    )
    verify_remote(repo_id, source_dir)
    print(f"[PASS] pushed {total / 1e9:.1f} GB to {repo_id}")
    return {"card": card, "files": files, "bytes": total}


def verify_remote(repo_id: str, source_dir: Path) -> None:
    """Confirm the Hub actually holds every local file at its local size.

    Check the remote listing independently of the uploader's return status.
    """
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id, repo_type="dataset", files_metadata=True)
    remote = {s.rfilename: (s.size or 0) for s in info.siblings}
    problems = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        local = path.stat().st_size
        if path.name not in remote:
            problems.append(f"{path.name} is absent from {repo_id}")
        elif remote[path.name] != local:
            problems.append(
                f"{path.name} is {remote[path.name]} bytes on the Hub, {local} locally"
            )
    if problems:
        raise RuntimeError(
            f"Upload to {repo_id} did not land: "
            + "; ".join(problems)
            + f". If the repo was recreated, delete {source_dir / '.cache'} so "
            "upload_large_folder re-hashes instead of resuming."
        )


def fetch(
    *,
    repo_id: str,
    dest_dir: Path,
    persist_id: str,
    body_names: list[str] | None = None,
    anchor_body: str | None = None,
    expected_motions: int | None = None,
    expected_transitions: int | None = None,
) -> Path:
    """Download into ``dest_dir`` and refuse anything that does not validate."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id, repo_type="dataset", local_dir=str(dest_dir), max_workers=8
    )
    _gate(
        dest_dir,
        persist_id=persist_id,
        body_names=body_names,
        anchor_body=anchor_body,
        expected_motions=expected_motions,
        expected_transitions=expected_transitions,
        what="use downloaded",
    )
    sidecar = _read_sidecar(dest_dir)
    print(
        f"[PASS] {dest_dir}: "
        f"{len(sidecar['traj_info']['ordered_traj_list']):,} trajectories, "
        f"{sidecar['traj_info']['written']:,} transitions"
    )
    return dest_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--repo-id", required=True)
        p.add_argument("--persist-id", required=True)
        p.add_argument("--body-names", nargs="+", default=None)
        p.add_argument("--anchor-body", default=None)
        p.add_argument("--expected-motions", type=int, default=None)
        p.add_argument("--expected-transitions", type=int, default=None)

    p_push = sub.add_parser("push", help="Validate and upload authorized arrays.")
    p_push.add_argument("--source-dir", type=Path, required=True)
    p_push.add_argument(
        "--source-repo",
        default=None,
        help="Source NPZ dataset repo, recorded in the generated card.",
    )
    p_push.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make the dataset public when the source terms permit public redistribution.",
    )
    p_push.add_argument("--dry-run", action="store_true", default=False)
    common(p_push)

    p_fetch = sub.add_parser("fetch", help="Download and validate on a compute node.")
    p_fetch.add_argument("--dest-dir", type=Path, required=True)
    common(p_fetch)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "push":
            push(
                source_dir=args.source_dir,
                repo_id=args.repo_id,
                persist_id=args.persist_id,
                body_names=list(args.body_names) if args.body_names else None,
                anchor_body=args.anchor_body,
                expected_motions=args.expected_motions,
                expected_transitions=args.expected_transitions,
                private=not args.public,
                source_repo=args.source_repo,
                dry_run=args.dry_run,
            )
        else:
            fetch(
                repo_id=args.repo_id,
                dest_dir=args.dest_dir,
                persist_id=args.persist_id,
                body_names=list(args.body_names) if args.body_names else None,
                anchor_body=args.anchor_body,
                expected_motions=args.expected_motions,
                expected_transitions=args.expected_transitions,
            )
    except RuntimeError as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
