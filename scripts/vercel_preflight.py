"""Pre-import checks for the Vercel container function.

The image installs its dependencies with `pip install --user`, which puts them
in the *user site* directory. CPython derives that directory from HOME when it
starts, and adds it to `sys.path` only if it resolves to the same place the
install used. Vercel wraps the image's CMD in its own entrypoint and does not
necessarily preserve the environment the image set -- PATH was the first
casualty, HOME is the second -- so the interpreter came up without that
directory on its path and `import fastapi` failed with the application never
having been reached.

The launcher now puts the directory on PYTHONPATH explicitly, using the path
resolved during the build rather than a guess. This script reports whether that
actually worked, and fails with the specific module name if it did not, so the
next failure is a sentence rather than a silent exit.

Runs before app.main is imported. Prints no environment variable values.
"""

from __future__ import annotations

import importlib
import os
import sys

# Deliberately cheap. This used to import torch and sentence-transformers too,
# which cost about six seconds before uvicorn could bind -- and binding fast is
# now the whole point, since the platform gives the container roughly fifteen
# seconds to accept a connection. The heavy stack is still proven importable,
# but at build time under `env -i`, where its cost is paid once instead of on
# every cold start. What remains here is the check that the import path itself
# works, which is what actually broke.
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
