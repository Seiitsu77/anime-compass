FROM python:3.12-slim

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    APP_HOST=0.0.0.0 \
    APP_PORT=7860 \
    EMBEDDING_PROVIDER=sentence_transformers \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION} \
    EMBEDDING_DIMENSIONS=384 \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_LOCAL_FILES_ONLY=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app
COPY --chown=user . $HOME/app
USER user
RUN pip install --no-cache-dir --user -r requirements-web.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', revision='${EMBEDDING_MODEL_REVISION}', device='cpu', trust_remote_code=False)"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/ready', timeout=3)" || exit 1
CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "7860"]
