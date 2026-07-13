"""The README's own structure — links, contents, heading ladder.

`test_public_api` checks that the README *mentions* every public name. This checks
that the document itself holds together: a link to a heading that was renamed, or a
new section missing from the Contents, is exactly the rot nobody notices by reading.
"""

import pathlib
import re
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8")
LINES = README.splitlines()


def project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def slug(text: str) -> str:
    """GitHub's heading slug: lowercase, strip punctuation, each space -> one hyphen.

    Runs of spaces are NOT collapsed, which is why "Training & Evaluation" becomes
    `training--evaluation` — the removed `&` leaves two spaces behind.
    """
    s = re.sub(r"[^\w\s-]", "", text.strip().lower())
    return s.replace(" ", "-")


def headings() -> list[tuple[int, int, str]]:
    """(level, line_no, text) for every heading outside a fenced code block."""
    found: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(LINES, 1):
        if re.match(r"^\s*```", line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if m := re.match(r"^(#{1,6})\s+(.*)$", line):
            found.append((len(m.group(1)), i, m.group(2)))
    return found


def internal_links() -> list[tuple[int, str]]:
    return [
        (i, m.group(1))
        for i, line in enumerate(LINES, 1)
        for m in re.finditer(r"\[[^\]]*\]\(#([^)]+)\)", line)
    ]


HEADINGS = headings()
SLUGS = [slug(text) for _, _, text in HEADINGS]


@pytest.mark.parametrize(("line", "anchor"), internal_links())
def test_every_internal_link_resolves(line, anchor):
    assert anchor in SLUGS, f"README:{line} links to #{anchor}, which is not a heading"


def test_no_two_headings_share_a_slug():
    dupes = {s for s in SLUGS if SLUGS.count(s) > 1}
    assert not dupes, f"duplicate heading slugs make their anchors ambiguous: {dupes}"


def test_the_heading_ladder_never_skips_a_level():
    previous = 0
    for level, line, text in HEADINGS:
        if previous:
            assert level <= previous + 1, f"README:{line} jumps to h{level}: {text!r}"
        previous = level


def test_every_section_is_listed_in_the_contents():
    linked = {anchor for _, anchor in internal_links()}
    missing = [
        text for level, _, text in HEADINGS
        if level > 1 and slug(text) not in linked
    ]
    assert not missing, f"sections absent from the Contents: {missing}"


def test_there_is_exactly_one_h1():
    assert sum(1 for level, _, _ in HEADINGS if level == 1) == 1


# ── Badges ────────────────────────────────────────────────────────────────────
# The badges are hand-written copies of pyproject's facts — the one duplication a README
# cannot avoid while the package is installed from git rather than PyPI. Pinned here, so a
# release that bumps the version cannot quietly leave the badge behind.


def test_the_version_badge_matches_pyproject():
    version = project_metadata()["version"]
    assert f"version-{version}-" in README, f"the version badge does not say {version}"


def test_the_python_badge_matches_requires_python():
    minimum = project_metadata()["requires-python"].lstrip(">=")
    # %E2%89%A5 is the "≥" the shield renders.
    assert f"python-%E2%89%A5{minimum}-" in README, f"the Python badge does not say ≥{minimum}"


def test_the_pytorch_badge_matches_the_torch_dependency():
    torch = next(d for d in project_metadata()["dependencies"] if d.startswith("torch"))
    minimum = torch.split(">=")[1]
    assert f"pytorch-%E2%89%A5{minimum}-" in README, f"the PyTorch badge does not say ≥{minimum}"
