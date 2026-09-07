#!/bin/sh
# Launcher for the Vercel container function.
#
# Vercel wraps the image's CMD in its own certificate entrypoint, and that
# wrapper does not necessarily preserve the image's PATH. The previous CMD --
#
#     CMD ["sh", "-c", "exec python run_app.py --host 0.0.0.0 --port ${PORT:-8000}"]
#
# -- resolved three things through PATH: `sh`, `python`, and `run_app.py`
# relative to WORKDIR. If PATH is not what the image set, the exec fails before
# the interpreter starts, which produces an exit status and no output at all --
# no traceback, no application log line. That is the failure being investigated.
#
# So nothing here is resolved through PATH. __PYTHON_BIN__ is replaced during
# the build with the interpreter's real location, taken from `sys.executable`
# in the final base image rather than assumed, and uvicorn is started as a
# module rather than through its console script, which lives in a bin directory
# the build already warned was off PATH.
#
# The [BOOT] markers exist to bisect a silent startup: each one that appears
# rules out everything before it. They are temporary diagnostics.
#
# Nothing here prints an environment variable other than PORT.

set -eu

echo "[BOOT] launcher started"

PY="__PYTHON_BIN__"
echo "[BOOT] python executable ${PY}"
if [ ! -x "${PY}" ]; then
    echo "[BOOT] FATAL: no executable interpreter at ${PY}"
    exit 1
fi

APP_DIR="__APP_DIR__"
if [ ! -d "${APP_DIR}" ]; then
    echo "[BOOT] FATAL: application directory ${APP_DIR} is missing"
    exit 1
fi
cd "${APP_DIR}"

# Required at runtime: without these the application cannot serve at all.
missing=""
for required in \
    app/main.py \
    frontend/index.html \
    data/artifacts.manifest.json \
    data/processed/anime_catalog.json \
    data/processed/collaborative_embeddings.npz
do
    if [ ! -f "${APP_DIR}/${required}" ]; then
        missing="${missing} ${required}"
    fi
done
if [ -n "${missing}" ]; then
    echo "[BOOT] FATAL: missing required files:${missing}"
    exit 1
fi

# Optional channels. Their absence degrades the recommender rather than
# stopping it, so it is reported and not treated as fatal.
for optional in \
    data/processed/semantic_embeddings.npz \
    data/processed/als_production_item_factors.npz \
    data/processed/reranker_features.npz \
    data/processed/reranker_lambdamart.txt
do
    if [ ! -f "${APP_DIR}/${optional}" ]; then
        echo "[BOOT] WARNING: optional artifact absent: ${optional}"
    fi
done

echo "[BOOT] artifacts present"

# Importing the app is cheap -- the expensive loading happens in the lifespan,
# once uvicorn starts it -- so this costs little and turns an import failure
# into a labelled error instead of a silent exit.
echo "[BOOT] importing app.main"
if ! "${PY}" -c "import app.main"; then
    echo "[BOOT] FATAL: app.main failed to import"
    exit 1
fi
echo "[BOOT] app.main import OK"

echo "[BOOT] starting uvicorn on port ${PORT:-80}"
exec "${PY}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-80}"
