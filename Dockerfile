# RegLens Maroc: API (default) or Streamlit interface (EXTRAS="llm,ui").
# Build:  docker build -t reglens .
#         docker build -t reglens-ui --build-arg EXTRAS=llm,ui .

FROM python:3.11-slim-bookworm AS builder
ARG EXTRAS=api,llm
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python --no-cache ".[${EXTRAS}]"

FROM python:3.11-slim-bookworm AS runtime
# Tesseract (French + Arabic) OCRs scanned PDFs uploaded through /v1/ingest.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 reglens
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=reglens:reglens ui ./ui
COPY --chown=reglens:reglens .streamlit ./.streamlit
RUN mkdir -p /data/raw && chown -R reglens:reglens /data
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    REGLENS_API_HOST=0.0.0.0 \
    REGLENS_LOG_JSON=true \
    REGLENS_DATA_DIR=/data/raw
USER reglens
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["reglens", "serve"]
