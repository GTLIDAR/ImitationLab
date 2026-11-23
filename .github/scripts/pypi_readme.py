"""Rewrite README.md with absolute links for the PyPI project page.

PyPI renders the package description outside the repository, where relative
links and images in ``README.md`` do not resolve. The release workflow runs
this script on its checkout before it builds the distributions:

    python .github/scripts/pypi_readme.py --ref v0.1.1

A link to a file that does not exist stops the script.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPOSITORY = "GTLIDAR/ImitationLab"

_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*:")
_IMAGE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
_ATTRIBUTE = re.compile(r'(\b(?:src|srcset)=")([^"]+)(")')
_LINK = re.compile(r"(\]\()([^)\s]+)(\))")
_FENCE = re.compile(r"\s*(```|~~~)")


def absolute(target: str, ref: str, *, image: bool) -> str:
    """Return ``target``, a link written in README.md, as an absolute URL."""
    if _SCHEME.match(target) or target.startswith("//"):
        return target
    if target.startswith("#"):
        return f"https://github.com/{REPOSITORY}{target}"
    path, hash_mark, anchor = target.partition("#")
    resolved = posixpath.normpath(path)
    if resolved.startswith("..") or not (REPO / resolved).exists():
        raise FileNotFoundError(f"README.md: link target not found: {target}")
    if image:
        return f"https://raw.githubusercontent.com/{REPOSITORY}/{ref}/{resolved}"
    kind = "tree" if (REPO / resolved).is_dir() else "blob"
    return f"https://github.com/{REPOSITORY}/{kind}/{ref}/{resolved}{hash_mark}{anchor}"


def rewrite(text: str, ref: str) -> str:
    """Make every relative link and image outside code blocks absolute."""

    def image(match: re.Match[str]) -> str:
        return (
            match.group(1) + absolute(match.group(2), ref, image=True) + match.group(3)
        )

    def link(match: re.Match[str]) -> str:
        return (
            match.group(1) + absolute(match.group(2), ref, image=False) + match.group(3)
        )

    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker and fence in (None, marker.group(1)):
            fence = None if fence else marker.group(1)
        elif fence is None:
            # Images first: afterwards their targets are absolute, and the
            # link pattern leaves them alone.
            line = _LINK.sub(link, _ATTRIBUTE.sub(image, _IMAGE.sub(image, line)))
        lines.append(line)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="Git tag or branch to link to.")
    parser.add_argument(
        "--output", type=Path, default=REPO / "README.md", help="File to write."
    )
    args = parser.parse_args()
    try:
        text = rewrite((REPO / "README.md").read_text(encoding="utf-8"), args.ref)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
