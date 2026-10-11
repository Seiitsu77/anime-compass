"""Check production imports before starting the web server.

The launcher supplies the resolved user site directory explicitly. Report
import failures without secrets; validate heavyweight ML imports at build time.
"""

from __future__ import annotations

import importlib
import os
import sys

# Only check lightweight imports here; the image build verifies ML packages.
REQUIRED_MODULES = (
    "fastapi",
    "numpy",
)


def main() -> int:
    expected = sys.argv[1] if len(sys.argv) > 1 else ""

    # Resolved comparison, not string equality: the same directory reaches
    # sys.path in a different spelling than PYTHONPATH carried it, and a
    # diagnostic that reports False while the imports succeed is worse than no
    # diagnostic at all.
    resolved = os.path.realpath(expected) if expected else ""
    on_sys_path = any(entry and os.path.realpath(entry) == resolved for entry in sys.path)

    print(f"[BOOT] python executable {sys.executable}")
    print(f"[BOOT] expected site-packages {expected}")
    print(f"[BOOT] site-packages exists {os.path.isdir(expected)}")
    print(f"[BOOT] site-packages on sys.path {on_sys_path}")

    failed: list[tuple[str, str]] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the reason is the diagnostic
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    if failed:
        for name, reason in failed:
            print(f"[BOOT] FATAL: cannot import {name}: {reason}")
        # sys.path is the thing that explains an import failure, and it holds
        # no secrets: it is a list of directories.
        print("[BOOT] sys.path:")
        for entry in sys.path:
            print(f"[BOOT]   {entry or '(cwd)'}")
        return 1

    print("[BOOT] runtime dependencies import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
