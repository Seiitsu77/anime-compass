"""Local backend for browser end-to-end checks of the React frontend.

Serves the real catalog, the production ALS artifact, and the LambdaMART
reranker, with three differences from a normal run:

* fast-path debug diagnostics are on, so a Like can be traced to the folded-in
  ALS user vector and the candidate ranking;
* sessions go to a throwaway SQLite file, never `data/anime_compass.db`;
* the Gemini key is blanked, so the agent answers from its deterministic
  fallback and nothing leaves this machine.

    python scripts/dev_e2e_backend.py            # http://127.0.0.1:8011
    python scripts/dev_e2e_backend.py --no-debug # production-like diagnostics

Point Vite at it with `API_PROXY_TARGET=http://127.0.0.1:8011` in
`frontend/.env.local`. Never use this entry point for a deployment.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--no-debug", action="store_true", help="leave debug diagnostics off, as in production")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    database = Path(tempfile.gettempdir()) / "anime_compass_e2e_sessions.db"
    os.environ["RECOMMENDATION_DEBUG_DIAGNOSTICS"] = "false" if args.no_debug else "true"
    os.environ["RATE_LIMIT_REQUESTS"] = "5000"
    os.environ["DATABASE_URL"] = f"sqlite:///{database.as_posix()}"
    os.environ["GEMINI_API_KEY"] = ""

    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
