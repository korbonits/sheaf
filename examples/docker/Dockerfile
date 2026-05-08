# Example user Dockerfile — extends the official sheaf-serve base image.
#
# The base at ghcr.io/korbonits/sheaf-serve:vX.Y.Z installs sheaf-serve
# core only.  This file shows the canonical pattern for adding the
# backend extras you need + your own server script.
#
# Build (from this directory, so `COPY server.py .` resolves):
#     cd examples/docker
#     docker build -t my-sheaf:dev .
#
# Run (CPU; bind to all interfaces inside the container so :8000
# is reachable from the host):
#     docker run --rm -p 8000:8000 my-sheaf:dev
#
# Then in another terminal:
#     curl -s http://localhost:8000/chronos/health
#     curl -s -X POST http://localhost:8000/chronos/predict \
#         -H 'Content-Type: application/json' \
#         -d '{
#               "model_type": "time_series",
#               "model_name": "chronos",
#               "history": [1.0, 2.0, 3.0, 4.0, 5.0],
#               "horizon": 3,
#               "frequency": "1h"
#             }' | jq .

# Pin to a specific sheaf-serve release.  Bump intentionally; don't track
# `:latest` in production.
FROM ghcr.io/korbonits/sheaf-serve:v0.9.0

# Add the backend extras your deployment actually needs.  Multiple extras
# in a single bracket expression keep the install in one resolver pass.
RUN pip install --no-cache-dir 'sheaf-serve[time-series]==0.9.0'

# Copy the server script that calls ModelServer.run().  /workspace is the
# image's WORKDIR, so this lands at /workspace/server.py.
COPY server.py .

# Run inside the venv that the base image set on PATH.  Binds 0.0.0.0
# inside the container so the published port (-p 8000:8000) reaches it.
CMD ["python", "server.py"]
