# Container image for KubeRay deployment.
#
# Extends the official sheaf-serve base, adds the time-series extra
# (Chronos / TimesFM / Moirai), and bakes app.py into /workspace where
# Ray Serve's import_path can find it.
#
# Build & push to your registry:
#     docker build -f examples/k8s/Dockerfile -t YOUR_REGISTRY/sheaf-chronos:v0.9.0 .
#     docker push YOUR_REGISTRY/sheaf-chronos:v0.9.0
#
# Then update the image: field in rayservice.yaml to match.

FROM ghcr.io/korbonits/sheaf-serve:v0.9.0

RUN pip install --no-cache-dir 'sheaf-serve[time-series]==0.9.0'

# /workspace is the base image's WORKDIR.  Ray Serve's import_path
# resolution does importlib.import_module("app"), which finds app.py on
# the worker's PYTHONPATH.  /workspace is on PYTHONPATH below.
COPY examples/k8s/app.py .

# Ray Serve workers inherit the head's environment.  PYTHONPATH=/workspace
# guarantees `import app` resolves regardless of which directory the
# worker process is launched from.
ENV PYTHONPATH=/workspace
