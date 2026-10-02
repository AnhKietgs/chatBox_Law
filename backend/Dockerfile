# syntax=docker/dockerfile:1.7
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Layer 1: ML libs (nặng, ít thay đổi) ────────────────────────────────
# Resolve embedding and document/OCR dependencies separately. FlagEmbedding
# pulls ir-datasets, whose loose ranges otherwise make pip explore a huge
# cross-product with Docling/PaddleOCR and fail with ResolutionTooDeep.
COPY requirements-embedding.txt requirements-docling.txt requirements-ocr.txt constraints-torch-cpu.txt ./
# The cache mount is owned by Docker BuildKit, not copied into the image. It
# makes a retried/invalidated ML layer reuse already downloaded wheels.
# Docker Desktop is not given a GPU in this local deployment. Install the CPU
# wheel first so FlagEmbedding's broad ``torch>=1.6`` dependency is already
# satisfied and pip does not download CUDA/CuDNN packages from PyPI.
RUN --mount=type=cache,id=lawrag-pip-cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.6.0+cpu" "torchvision==0.21.0+cpu"
RUN --mount=type=cache,id=lawrag-pip-cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints-torch-cpu.txt -r requirements-embedding.txt
RUN --mount=type=cache,id=lawrag-pip-cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints-torch-cpu.txt -r requirements-docling.txt
RUN --mount=type=cache,id=lawrag-pip-cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints-torch-cpu.txt -r requirements-ocr.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 2: App deps (nhẹ, hay thay đổi) ───────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,id=lawrag-pip-cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip -r requirements.txt

# ── Layer 3: Source code ─────────────────────────────────────────────────
COPY app ./app
COPY tests ./tests
COPY pytest.ini .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
