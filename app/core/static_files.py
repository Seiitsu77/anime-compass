from __future__ import annotations

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class FrontendStaticFiles(StaticFiles):
    """Revalidate the entry page; cache Vite's content-hashed assets."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        path = path.replace("\\", "/")
        if response.status_code in {200, 304}:
            if path in {"", ".", "index.html"} or path.endswith(".html"):
                response.headers["Cache-Control"] = "no-cache"
            elif path.startswith("assets/"):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
