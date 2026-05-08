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
#
# Why no venv:
#     We deliberately install sheaf-serve into the system Python at
#     /usr/local/lib/python3.11/site-packages instead of a venv.  Containers
#     are already isolated; a venv adds a layer of indirection that fights
#     K8s tooling (KubeRay's injected `ray start` command does PATH lookup
#     and expects /usr/local/bin/ray, not /app/.venv/bin/ray).  System
#     install puts ray, python, pip, and any user-installed extras at the
#     conventional /usr/local/bin location, so derived images and
#     orchestrators don't need to know about a venv.

# ---------------------------------------------------------------------------
# Stage 1: builder — install sheaf-serve into the system Python via uv.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# UV_LINK_MODE=copy avoids a hardlink fallback warning when uv copies between
# its cache and the install target.  UV_COMPILE_BYTECODE=1 pre-compiles .pyc
# files at install time so first-request import isn't slower.
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

# --system installs into the system Python at /usr/local/lib/python3.11/
# site-packages and /usr/local/bin.  This is what the runtime stage copies
# wholesale; nothing references /app/.venv after this stage exits.
RUN if [ -n "$SHEAF_EXTRAS" ]; then \
        uv pip install --system --no-cache "sheaf-serve[${SHEAF_EXTRAS}]==${SHEAF_VERSION}"; \
    else \
        uv pip install --system --no-cache "sheaf-serve==${SHEAF_VERSION}"; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal Python slim image with the system-installed
# sheaf packages copied in from the builder stage.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONUNBUFFERED=1 ensures container logs flush in real time, otherwise
# Ray Serve / FastAPI logs buffer and disappear when the container exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime libs that some optional extras pull in transitively but
# expect to find on the system: libgomp1 (numpy / pyarrow / faiss).
# build-essential is left out — extras that need a compiler should add it
# themselves in a derived image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the system-Python install from the builder stage.  This pulls in:
#   - /usr/local/lib/python3.11/site-packages — every Python package sheaf
#     depends on (ray, fastapi, pydantic, httpx, numpy, prometheus_client, …)
#   - /usr/local/bin — every console script those packages ship (ray,
#     uvicorn, fastapi, etc.) plus pip / uv themselves.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# /workspace is the conventional mount point for user source.  Keeping
# WORKDIR there means `COPY . .` in derived images lands in a sensible
# place without the user having to set WORKDIR themselves.
WORKDIR /workspace

# 8000 is the default port `ModelServer.run()` listens on; expose it for
# documentation purposes (no `-p` flag is implied — users still need to
# publish the port at run time).
EXPOSE 8000

# Sanity check that sheaf-serve imports + ray binary resolves.  Fails the
# build if the install stage produced an unimportable environment or if
# the entry-point scripts didn't get copied across.
RUN python -c "import sheaf; print(f'sheaf {sheaf.__version__}')" \
    && ray --version

# No ENTRYPOINT or CMD — this is a base image.  Derived images set their
# own (typically `CMD ["python", "server.py"]`).
