"""Fail with an instruction, not a stack trace, when run by the wrong interpreter.

macOS has no bare ``python``, and its ``python3`` has none of this project's
dependencies — so the obvious command produces either "command not found" or an
ImportError three frames deep in someone else's module. Both are the same mistake
and both deserve the same one-line answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def require_venv(*modules: str) -> None:
    """Check we are running somewhere the dependencies exist; explain if not."""
    missing = []
    for name in modules:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return

    script = Path(sys.argv[0]).name
    relative = Path(sys.argv[0])
    try:
        relative = Path(sys.argv[0]).resolve().relative_to(ROOT)
    except ValueError:
        pass

    lines = [
        "",
        f"{script} needs {', '.join(missing)}, which this Python does not have.",
        f"  running: {sys.executable}",
        "",
    ]
    if VENV_PYTHON.exists():
        lines += [
            "Run it with the project's own Python instead:",
            f"    .venv/bin/python {relative}",
        ]
    else:
        lines += [
            "The project environment has not been created yet:",
            "    make setup",
            "",
            "then re-run:",
            f"    .venv/bin/python {relative}",
        ]
    lines.append("")
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)
