#!/bin/sh
# Launch production with paths resolved when the image was built.
# Diagnostics report dependencies and artifacts without printing secrets.

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

# Use the site-packages directory resolved during the image build.
SITE_PACKAGES="__SITE_PACKAGES__"
if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH="${SITE_PACKAGES}:${PYTHONPATH}"
else
    PYTHONPATH="${SITE_PACKAGES}"
fi
export PYTHONPATH

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

# Confirm the import path actually works before blaming the application for a
# dependency problem. This is what turned "No module named 'fastapi'" from a
# guess into a fact.
if ! "${PY}" "${APP_DIR}/scripts/container_preflight.py" "${SITE_PACKAGES}"; then
    echo "[BOOT] FATAL: runtime dependencies are not importable"
    exit 1
fi

# Importing the app is cheap -- model loading runs in a background worker --
# so this costs little and turns an import failure
# into a labelled error instead of a silent exit.
echo "[BOOT] importing app.main"
if ! "${PY}" -c "import app.main"; then
    echo "[BOOT] FATAL: app.main failed to import"
    exit 1
fi
echo "[BOOT] app.main import OK"

echo "[BOOT] starting uvicorn on port ${PORT:-8080}"
exec "${PY}" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
