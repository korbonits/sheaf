# syntax=docker/dockerfile:1.7
#
# Base image for Sheaf deployments — sheaf-serve core only.
#
# Users extend this image to:
#   1. Install backend extras (`sheaf-serve[time-series,vision,...]`)
#   2. COPY in their own server script (e.g. `server.py`)
#   3. Set CMD to run their ModelServer
#
# A worked example of that extension pattern lives at
# examples/docker/.  The official build is published on tag pushes to
# `ghcr.io/korbonits/sheaf-serve:vX.Y.Z` (and `:latest`) — see
# .github/workflows/docker.yml.
#
# Build locally:
#     docker build -t sheaf-serve:dev --build-arg SHEAF_VERSION=0.9.0 .
#
# CUDA-based GPU builds:
#     Swap the base in both stages for an NVIDIA CUDA image, e.g.
#     `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, and install Python 3.11.
#     Out of scope for the base image; the v0.10 README documents the
#     pattern.

# ---------------------------------------------------------------------------
# Stage 1: builder — uv installs sheaf-serve into a clean venv at /app/.venv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# UV_LINK_MODE=copy avoids a hardlink fallback warning when the venv is on a
# different filesystem than the global cache (which it is, since we copy the
# venv to the runtime stage).  UV_COMPILE_BYTECODE=1 pre-compiles .pyc files
# at install time so import isn't slower on first request.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Pin the sheaf-serve version baked into the image.  CI overrides this at
# build time per release tag (.github/workflows/docker.yml).  No default —
# building without explicit version is a footgun.
ARG SHEAF_VERSION

# Empty by default — users typically extend this image and add extras in
# their own Dockerfile.  Provide as a comma-separated list of extras names,
# e.g. SHEAF_EXTRAS="time-series,vision" for an image with those backends
# pre-installed.
ARG SHEAF_EXTRAS=""

RUN uv venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install sheaf-serve at the pinned version, optionally with extras.  The
# bracket expansion only runs when SHEAF_EXTRAS is non-empty.
RUN if [ -n "$SHEAF_EXTRAS" ]; then \
        uv pip install --no-cache "sheaf-serve[${SHEAF_EXTRAS}]==${SHEAF_VERSION}"; \
    else \
        uv pip install --no-cache "sheaf-serve==${SHEAF_VERSION}"; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal Python slim image with the prebuilt venv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONUNBUFFERED=1 ensures container logs flush in real time, otherwise
# Ray Serve / FastAPI logs buffer and disappear when the container exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install runtime libs that some optional extras pull in transitively but
# expect to find on the system: libgomp1 (numpy / pyarrow / faiss), and
# build-essential is left out — extras that need a compiler should add it
# themselves in a derived image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Drop the venv from the builder stage.  Nothing else from the builder
# leaks into the runtime image — keeps the layer small and the surface
# area predictable.
COPY --from=builder /app/.venv /app/.venv

# /workspace is the conventional mount point for user source.  Keeping
# WORKDIR there means `COPY . .` in derived images lands in a sensible
# place without the user having to set WORKDIR themselves.
WORKDIR /workspace

# 8000 is the default port `ModelServer.run()` listens on; expose it for
# documentation purposes (no `-p` flag is implied — users still need to
# publish the port at run time).
EXPOSE 8000

# Sanity check that sheaf-serve imports — fails the build if the install
# stage produced an unimportable environment.  Cheap; runs at build time.
RUN python -c "import sheaf; print(f'sheaf {sheaf.__version__}')"

# No ENTRYPOINT or CMD — this is a base image.  Derived images set their
# own (typically `CMD ["python", "server.py"]`).
