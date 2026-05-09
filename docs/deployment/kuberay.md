# KubeRay

[KubeRay](https://github.com/ray-project/kuberay) is the official
operator for Ray on Kubernetes. It defines a `RayService` CRD that
combines a Ray cluster spec with a Ray Serve config — exactly the
shape sheaf-serve wants.

The repo ships a complete reference manifest at
[`examples/k8s/`](https://github.com/korbonits/sheaf/tree/main/examples/k8s)
that this page walks through.

## Prerequisites

1. A Kubernetes cluster — local (Kind, Minikube), or hosted (EKS,
   GKE, AKS).
2. KubeRay operator:
   ```bash
   helm repo add kuberay https://ray-project.github.io/kuberay-helm/
   helm install kuberay-operator kuberay/kuberay-operator --version 1.2.2
   ```
3. A container image with sheaf-serve + your model extras + your
   server entrypoint, pushed to a registry your cluster can pull from.

## The container image

The KubeRay path needs an image that exposes a Ray Serve `Application`,
not a `ModelServer.run()` call — RayService deploys applications by
name via `serveConfigV2.applications[].import_path`. Use the public
`sheaf.build_app(spec)` helper:

```python title="app.py"
from sheaf import ModelSpec, build_app
from sheaf.api.base import ModelType

app = build_app(ModelSpec(
    name="chronos",
    model_type=ModelType.TIME_SERIES,
    backend="chronos2",
    backend_kwargs={
        "model_id": "amazon/chronos-bolt-tiny",
        "device_map": "cpu",
        "torch_dtype": "float32",
    },
))
```

```dockerfile title="Dockerfile"
FROM ghcr.io/korbonits/sheaf-serve:v0.10.0
RUN pip install --no-cache-dir 'sheaf-serve[time-series]==0.10.0'
COPY app.py .
```

Build and push:

```bash
cd examples/k8s
docker build -t YOUR_REGISTRY/sheaf-chronos:v0.10.0 .
docker push YOUR_REGISTRY/sheaf-chronos:v0.10.0
```

## The RayService manifest

Excerpt from [`examples/k8s/rayservice.yaml`](https://github.com/korbonits/sheaf/blob/main/examples/k8s/rayservice.yaml):

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: sheaf-chronos
spec:
  serveConfigV2: |
    applications:
      - name: chronos
        import_path: app:app
        route_prefix: /chronos

  rayClusterConfig:
    rayVersion: '2.10.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
      template:
        spec:
          containers:
            - name: ray-head
              image: YOUR_REGISTRY/sheaf-chronos:v0.10.0
              ports:
                - {containerPort: 8000, name: serve}
                - {containerPort: 8265, name: dashboard}
              resources:
                limits: {cpu: "2", memory: "4Gi"}
                requests: {cpu: "1", memory: "2Gi"}
```

The `serveConfigV2` is a standard Ray Serve config — multiple
applications go in the list, each with their own `import_path` and
`route_prefix`.

## Apply + verify

```bash
kubectl apply -f examples/k8s/rayservice.yaml
kubectl get rayservice sheaf-chronos -w
```

The `Running` state takes ~2 minutes on a fresh cluster — the head
pulls the image, Ray boots, the Serve application deploys, the
backend's `load()` runs.

```bash
kubectl port-forward svc/sheaf-chronos-serve-svc 8000:8000
curl http://localhost:8000/chronos/health
```

## Multi-model deployments

Add applications to the `serveConfigV2.applications` list. Each one
references its own `app.py` `app` attribute (or you can put multiple
in one file and reference different attributes). KubeRay deploys
each as an independent Serve application with its own scaling
profile.

## Worker groups (scale-out)

The reference manifest runs head-only — fine for CPU-bound, single-
replica deploys. For multi-replica or GPU workloads, uncomment the
`workerGroupSpecs` block at the bottom of the manifest, set
`replicas`, `minReplicas`, `maxReplicas`, and (for GPU) a node
selector + GPU resource request + a CUDA-based image.

## Why KubeRay over plain Deployments

A naive `kind: Deployment` running `python app.py` works for one
replica. The moment you want **(a)** multi-replica with shared scaling
state, **(b)** declarative model rollout/rollback, or **(c)** the Ray
dashboard and observability surface, KubeRay's `RayService` is the
shorter path. It owns the cluster lifecycle and treats the Serve
config as part of the spec — `kubectl apply` triggers a
zero-downtime rolling deploy of the new Serve application alongside
the old one, identical to what `ModelServer.update(spec)` does
locally.

## Reference

- Manifest: [`examples/k8s/rayservice.yaml`](https://github.com/korbonits/sheaf/blob/main/examples/k8s/rayservice.yaml)
- Image source: [`examples/k8s/Dockerfile`](https://github.com/korbonits/sheaf/blob/main/examples/k8s/Dockerfile) + [`examples/k8s/app.py`](https://github.com/korbonits/sheaf/blob/main/examples/k8s/app.py)
- KubeRay docs: [ray-project/kuberay](https://github.com/ray-project/kuberay)
