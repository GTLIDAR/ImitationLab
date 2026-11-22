"""Check the version and bundled notices in both release distributions."""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check_dist(directory: Path, tag: str | None) -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    version = project["version"]
    if tag is not None and tag != f"v{version}":
        raise ValueError(
            f"Release tag {tag!r} does not match package version {version}"
        )
    # Verify the actual bytes at their install locations, including licenses
    # referenced by the source files that ship in the wheel.
    notices = [
        Path("LICENSE"),
        Path("THIRD_PARTY_NOTICES.md"),
        *sorted(Path("third_party/licenses").glob("*.txt")),
    ]
    stem = f"{project['name']}-{version}"
    with (
        tarfile.open(directory / f"{stem}.tar.gz") as source,
        zipfile.ZipFile(directory / f"{stem}-py3-none-any.whl") as wheel,
    ):

        def read_source(name: str) -> bytes:
            entry = source.extractfile(f"{stem}/{name}")
            if entry is None:
                raise ValueError(f"Missing source distribution file {name}")
            return entry.read()

        for metadata in (
            read_source("PKG-INFO"),
            wheel.read(f"{stem}.dist-info/METADATA"),
        ):
            if BytesParser().parsebytes(metadata)["Version"] != version:
                raise ValueError(
                    "Distribution metadata version differs from the project version"
                )
        for path in notices:
            expected = path.read_bytes()
            if read_source(path.as_posix()) != expected:
                raise ValueError(f"Incorrect {path} in source distribution")
            if wheel.read(f"{stem}.dist-info/licenses/{path.as_posix()}") != expected:
                raise ValueError(f"Incorrect {path} in wheel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    check_dist(args.directory, args.tag)
