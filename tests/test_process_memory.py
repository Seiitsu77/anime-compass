"""Process-memory measurement must work on every platform CI runs on.

Four copies of a Windows-only PSAPI block had accumulated across the evaluation
harness and one script. They type-checked on the developer machine and failed
CI on Linux, because `ctypes.windll` exists only on Windows. The copies are the
reason it took three files to fix, so the guard below is as much about keeping
them consolidated as about the measurement itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

from backend.anime_agent import process_memory
from backend.anime_agent.process_memory import (
    current_process_rss_bytes,
    peak_process_rss_bytes,
)

SOURCE_ROOTS = (Path("app"), Path("backend"), Path("scripts"))
OWNER = Path("backend/anime_agent/process_memory.py")


def test_the_helpers_answer_on_this_platform():
    """Whatever platform runs the tests, these must return usable numbers."""
    for value in (peak_process_rss_bytes(), current_process_rss_bytes()):
        assert value is not None, "this platform should be able to report resident memory"
        assert value > 0


def test_a_missing_measurement_is_none_rather_than_an_exception():
    """These numbers annotate manifests; they must never fail a run."""
    assert process_memory._windows_working_set() is None or isinstance(process_memory._windows_working_set(), tuple)


def windll_references(path: Path) -> list[int]:
    """Line numbers where `ctypes.windll` is accessed in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "windll"
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctypes"
    ]


def test_only_one_module_touches_the_windows_api():
    """Anywhere else, `ctypes.windll` breaks type-checking and CI on Linux."""
    offenders: dict[str, list[int]] = {}
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path == OWNER or OWNER.name in path.parts:
                continue
            lines = windll_references(path)
            if lines:
                offenders[str(path)] = lines
    assert not offenders, (
        f"ctypes.windll must live only in {OWNER}, which guards it behind a "
        f"sys.platform check; found it in: {offenders}"
    )


def test_the_owner_guards_every_platform_branch():
    """The guard is a sys.platform comparison, which mypy narrows on.

    A runtime-only check such as `os.name == "nt"` would still be type-checked
    for Linux, which is exactly how the original failure got through.
    """
    source = OWNER.read_text(encoding="utf-8")
    assert 'sys.platform != "win32"' in source
    assert 'sys.platform == "win32"' in source
