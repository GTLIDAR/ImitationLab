"""Build the documentation site from the Markdown guides.

The guides are plain Markdown that reads well on GitHub, and some of them live
next to the code they describe. This script stages them in ``docs/_build/src``
together with the landing page, rewrites relative links for the staged layout,
and runs Sphinx on the staged tree:

    pixi run -e docs docs-build

A link to a file that is not part of the site becomes a link to that file on
GitHub. A link to a file that does not exist stops the build.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SITE = Path(__file__).resolve().parent
BUILD = REPO / "docs" / "_build"

REPOSITORY_URL = "https://github.com/GTLIDAR/ImitationLab"

# Guides kept outside ``docs/``, by repository path, and their page in the site.
EXTERNAL_PAGES = {
    "data/README.md": "motion-data.md",
    "imitationlab/algorithms/README.md": "policy-learners.md",
    "imitationlab/data_tools/README.md": "motion-data-tools.md",
    "CONTRIBUTING.md": "contributing.md",
    "CHANGELOG.md": "changelog.md",
    "SECURITY.md": "security.md",
    "THIRD_PARTY_NOTICES.md": "third-party-notices.md",
}

# The site has its own landing page; the GitHub index of docs/ is not staged.
LANDING_PAGE = "docs/_site/index.md"
GITHUB_INDEX = "docs/README.md"

_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?((?:\s+\"[^\"]*\")?\s*)\)")
_CODE_SPAN = re.compile(r"(`+[^`]*`+)")
_SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*:")
_FENCE = re.compile(r"\s*(```|~~~)")


class BrokenLink(ValueError):
    """A relative link whose target is not in the repository."""


def repository_files() -> set[str]:
    """Tracked files plus new files that Git does not ignore."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line for line in listing.stdout.splitlines() if (REPO / line).is_file()}


def site_pages(files: set[str]) -> dict[str, str]:
    """Map each staged file's repository path to its path in the site."""
    pages = dict(EXTERNAL_PAGES)
    pages[LANDING_PAGE] = "index.md"
    for path in files:
        if not path.startswith("docs/") or path.startswith("docs/_site/"):
            continue
        if path != GITHUB_INDEX:
            pages[path] = path.removeprefix("docs/")
    missing = sorted(set(pages) - files)
    if missing:
        raise FileNotFoundError(f"site pages missing from the repository: {missing}")
    return pages


def rewrite_target(
    target: str, source: str, pages: dict[str, str], files: set[str], ref: str
) -> str:
    """Return ``target``, a link written in ``source``, as the site needs it."""
    if _SCHEME.match(target) or target.startswith(("#", "//")):
        return target
    path, hash_mark, anchor = target.partition("#")
    fragment = hash_mark + anchor
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if resolved.startswith(".."):
        raise BrokenLink(f"{source}: link leaves the repository: {target}")
    here = posixpath.dirname(pages[source]) or "."
    if resolved in pages:
        return posixpath.relpath(pages[resolved], here) + fragment
    if resolved in (GITHUB_INDEX, "docs"):
        return posixpath.relpath("index.md", here)
    if resolved == "README.md":
        return REPOSITORY_URL + (fragment or "#readme")
    if resolved in files:
        return f"{REPOSITORY_URL}/blob/{ref}/{resolved}{fragment}"
    if resolved == "." or any(name.startswith(resolved + "/") for name in files):
        return f"{REPOSITORY_URL}/tree/{ref}/{resolved}".removesuffix("/.")
    raise BrokenLink(f"{source}: link target not found: {target}")


def rewrite_links(
    text: str, source: str, pages: dict[str, str], files: set[str], ref: str
) -> str:
    """Rewrite every relative Markdown link outside code in ``text``."""

    def replace(match: re.Match[str]) -> str:
        target = rewrite_target(match.group(1), source, pages, files, ref)
        return f"]({target}{match.group(2)})"

    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if marker and fence in (None, marker.group(1)):
            fence = None if fence else marker.group(1)
        elif fence is None:
            parts = _CODE_SPAN.split(line)
            parts[::2] = [_LINK.sub(replace, part) for part in parts[::2]]
            line = "".join(parts)
        lines.append(line)
    return "".join(lines)


def navigation_pages(landing: str) -> set[str]:
    """Pages named in the landing page's tables of contents."""
    named = set()
    for block in re.findall(r"```\{toctree\}\n(.*?)```", landing, flags=re.DOTALL):
        for entry in block.splitlines():
            entry = entry.strip()
            if entry and not entry.startswith(":"):
                # Either "page" or "Title <page>".
                titled = re.search(r"<(.+)>$", entry)
                named.add((titled.group(1) if titled else entry) + ".md")
    return named


def stage(stage_dir: Path, ref: str) -> None:
    """Write the site sources into ``stage_dir``."""
    files = repository_files()
    pages = site_pages(files)
    landing = (REPO / LANDING_PAGE).read_text(encoding="utf-8")
    in_navigation = navigation_pages(landing) | {"index.md"}

    shutil.rmtree(stage_dir, ignore_errors=True)
    for source, page in sorted(pages.items()):
        destination = stage_dir / page
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not page.endswith(".md"):
            shutil.copyfile(REPO / source, destination)
            continue
        text = (REPO / source).read_text(encoding="utf-8")
        if source != LANDING_PAGE:
            text = rewrite_links(text, source, pages, files, ref)
        if page not in in_navigation:
            # Reached by links from a report, not from the navigation.
            text = "---\norphan: true\n---\n\n" + text
        destination.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--builder", default="html", help="Sphinx builder, for example linkcheck."
    )
    parser.add_argument(
        "--ref", default="main", help="Git ref for links to files on GitHub."
    )
    args = parser.parse_args()

    stage_dir = BUILD / "src"
    try:
        stage(stage_dir, args.ref)
    except (BrokenLink, FileNotFoundError) as error:
        print(f"docs: {error}", file=sys.stderr)
        return 1

    from sphinx.cmd.build import main as sphinx_build

    return sphinx_build(
        [
            "--builder",
            args.builder,
            "--conf-dir",
            str(SITE),
            "--fail-on-warning",
            "--keep-going",
            "--nitpicky",
            "--quiet",
            "--doctree-dir",
            str(BUILD / "doctrees"),
            str(stage_dir),
            str(BUILD / args.builder),
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
