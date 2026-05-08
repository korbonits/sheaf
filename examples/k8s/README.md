# Sheaf on Kubernetes via KubeRay

This directory shows how to deploy a sheaf-serve `ModelSpec` to a Kubernetes
cluster using KubeRay's `RayService` resource.  The pattern mirrors what
`ModelServer.run()` does locally: same `ModelSpec`, same Ray Serve
Application — `RayService` is just the K8s-native way to keep that
Application running.

## Files

- **`app.py`** — calls `sheaf.build_app(spec)` to produce a Ray Serve
  Application.  Edit the `ModelSpec` here to swap backends, add resources,
  attach LoRA adapters, etc.
- **`Dockerfile`** — extends the official `ghcr.io/korbonits/sheaf-serve:vX.Y.Z`
  base image, installs the backend extra(s) you need, and bakes `app.py` into
  `/workspace`.
- **`rayservice.yaml`** — KubeRay `RayService` manifest.  References the
  image you build from the Dockerfile and points `import_path: app:app` at
  the Application defined in `app.py`.

## Prerequisites

1. **A Kubernetes cluster.**  Anything works — Minikube / Kind / Docker
   Desktop locally; EKS / GKE / AKS for prod.
2. **The KubeRay operator** installed in the cluster:
   ```bash
   helm repo add kuberay https://ray-project.github.io/kuberay-helm/
   helm install kuberay-operator kuberay/kuberay-operator --version 1.2.2
   ```
3. **A container registry your cluster can pull from.**  GHCR, ECR, GCR,
   Docker Hub — any of them.

## Deploy

### 1. Build and push the image

```bash
cd examples/k8s
docker build -t YOUR_REGISTRY/sheaf-chronos:v0.9.0 .
docker push YOUR_REGISTRY/sheaf-chronos:v0.9.0
```

The Dockerfile uses `COPY app.py .`, so the build context must be a
directory that has `app.py` at the root.  In production, copy this
`Dockerfile` and `app.py` into your own project root and run
`docker build .` there.

### 2. Update the manifest

In `rayservice.yaml`, replace `YOUR_REGISTRY/sheaf-chronos:v0.9.0` with
the path you just pushed.

### 3. Apply

```bash
kubectl apply -f examples/k8s/rayservice.yaml
kubectl get rayservice sheaf-chronos -w
```

The `Ready` column flips to `True` once the head pod is running and the
Ray Serve Application is healthy — typically ~2 minutes on a fresh
cluster while the image pulls.

### 4. Smoke test

```bash
kubectl port-forward svc/sheaf-chronos-serve-svc 8000:8000
```

In another terminal:

```bash
curl -s http://localhost:8000/chronos/health
# {"status":"ok"}

curl -s -X POST http://localhost:8000/chronos/predict \
    -H 'Content-Type: application/json' \
    -d '{
          "model_type": "time_series",
          "model_name": "chronos",
          "history": [312, 298, 275, 260, 255, 263, 285, 320,
                      368, 402, 421, 435, 442, 438, 430, 425],
          "horizon": 6,
          "frequency": "1h"
        }' | jq '.mean'
```

You can also reach the Ray dashboard at `http://localhost:8265` (after
`kubectl port-forward svc/sheaf-chronos-head-svc 8265:8265`).

## Scaling out

The example deploys a single head pod with no workers — sufficient for
chronos-bolt-tiny on CPU.  For larger workloads:

- Uncomment the `workerGroupSpecs` block in `rayservice.yaml`.  Set
  `replicas` (or use `minReplicas` / `maxReplicas` with
  `enableInTreeAutoscaling: true` on the cluster spec).
- For GPU workloads, swap the official CPU base for an NVIDIA CUDA
  runtime image in the `Dockerfile`, add `nvidia.com/gpu: 1` to the
  worker container's `resources`, and add a `nodeSelector` for your
  GPU node pool.

## Hot-swap deployments

`RayService` supports zero-downtime updates.  Edit `app.py` (e.g. point
to a different `model_id` or scale `num_replicas`), rebuild + push the
image, then bump the `image:` tag in the manifest and `kubectl apply`.
KubeRay performs a rolling update — new pods come up with the new spec
while old pods drain in-flight requests.

## Troubleshooting

- `kubectl describe rayservice sheaf-chronos` — top-level events.
- `kubectl logs -l ray.io/cluster=sheaf-chronos -c ray-head` — head pod logs.
- `kubectl get pods -l ray.io/cluster=sheaf-chronos` — pod status.
- If `Ready=False` persists, check `kubectl describe pod` on the head
  pod — image pull errors, resource constraints, and OOM are the usual
  suspects.
