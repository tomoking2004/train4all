"""What the project says about itself — the README's structure, the docstrings' shape,
and the facts restated outside pyproject.

`test_public_api` checks that the README *mentions* every public name. This checks that
those statements hold together: a link to a heading that was renamed, a new section
missing from the Contents, a docstring opened against the house style, a badge or a CI
matrix still repeating a fact pyproject has since changed — exactly the rot nobody
notices by reading.
"""

import ast
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


def test_the_pinned_install_example_matches_pyproject():
    """The badge is not the only place a version is written out by hand."""
    version = project_metadata()["version"]
    assert f"@v{version}\n" in README, f"the pinned install example does not say v{version}"


def test_the_python_badge_matches_requires_python():
    minimum = project_metadata()["requires-python"].lstrip(">=")
    # %E2%89%A5 is the "≥" the shield renders.
    assert f"python-%E2%89%A5{minimum}-" in README, f"the Python badge does not say ≥{minimum}"


def test_the_pytorch_badge_matches_the_torch_dependency():
    torch = next(d for d in project_metadata()["dependencies"] if d.startswith("torch"))
    minimum = torch.split(">=")[1]
    assert f"pytorch-%E2%89%A5{minimum}-" in README, f"the PyTorch badge does not say ≥{minimum}"


# ── CI ────────────────────────────────────────────────────────────────────────
# The workflow's matrix is the same kind of hand-written copy as the badges above: the
# classifiers promise which Python versions this package runs on, and the matrix is what
# would find out. A version added to one and not the other makes the promise untested.

PYTHON_CLASSIFIER = "Programming Language :: Python :: "


def test_the_ci_matrix_matches_the_python_classifiers():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix = re.search(r"python-version: \[(.*)\]", workflow)
    assert matrix, "the CI workflow declares no python-version matrix"

    tested = {v.strip().strip('"') for v in matrix.group(1).split(",")}
    classified = {
        c.removeprefix(PYTHON_CLASSIFIER)
        for c in project_metadata()["classifiers"]
        if c.startswith(PYTHON_CLASSIFIER)
    }
    # `:: 3` names the family, not a version anything could be run on.
    promised = {v for v in classified if "." in v}
    assert tested == promised, f"CI tests {sorted(tested)} but pyproject promises {sorted(promised)}"


# ── Docstrings ────────────────────────────────────────────────────────────────
# Two openings coexist in this project, and the choice between them is not free:
# `"""Summary…` while the docstring is one unbroken run of prose, `"""` alone on its line
# once that summary is merely the first of several parts. Nothing but habit had held the
# distinction across every docstring here, and habit is what a newcomer cannot read.

DEFINITION = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
SECTION = re.compile(r"^[ \t]*(?:Args|Attributes|Raises|Returns|Yields):[ \t]*$", re.M)
BULLET = re.compile(r"^[ \t]*[-*][ \t]", re.M)
HEADING_RULE = re.compile(r"^[ \t]*─{3,}[ \t]*$", re.M)


def docstring_literal(node: ast.AST) -> tuple[ast.Constant, str] | None:
    """The node a definition opens with and the text it carries, or None when it opens
    with something other than a string."""
    if not isinstance(node, DEFINITION) or not node.body:
        return None
    first = node.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
        return None
    # The text travels back with the node: an `ast.Constant` holds any literal, so
    # the caller would have to re-establish that this one is a string.
    text = first.value.value
    return (first.value, text) if isinstance(text, str) else None


def carries_structure(text: str) -> bool:
    """Whether a docstring holds more than prose: a section block, a bullet, a ruled heading."""
    return bool(SECTION.search(text) or BULLET.search(text) or HEADING_RULE.search(text))


def docstrings() -> list[tuple[str, bool, bool]]:
    """(location, opens on its own line, carries structure) for every docstring in the project."""
    found: list[tuple[str, bool, bool]] = []
    for path in sorted(p for d in ("train4all", "tests") for p in (ROOT / d).rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if (opening := docstring_literal(node)) is None:
                continue
            literal, text = opening
            raw = ast.get_source_segment(source, literal) or ""
            after_quotes = raw.partition('"""')[2] or raw.partition("'''")[2]
            found.append((
                f"{path.relative_to(ROOT).as_posix()}:{literal.lineno}",
                after_quotes.startswith("\n"),
                carries_structure(text),
            ))
    return found


@pytest.mark.parametrize(("location", "own_line", "structured"), docstrings())
def test_every_docstring_opens_the_way_its_shape_asks(location, own_line, structured):
    if structured:
        assert own_line, (
            f"{location}: a section block, a bullet list, or a ruled heading leaves the summary "
            f'one part among several — drop it below the opening """'
        )
    else:
        assert not own_line, (
            f'{location}: unbroken prose runs on from its summary — keep it on the opening """'
        )
