from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import uvicorn  # noqa: E402

from app.core.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Anime Compass FastAPI application.")
    parser.add_argument("--host", default=settings.app_host)
    parser.add_argument("--port", default=settings.app_port, type=int)
    args = parser.parse_args()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
