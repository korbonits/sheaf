# Deployment

Three production paths, picked by the operational profile that fits.

| Path | When to pick it |
|---|---|
| **[Docker](docker.md)** | Self-hosted box / VM, you own the machine. Quickest path from `pip install` to a real production deploy. |
| **[KubeRay](kuberay.md)** | Kubernetes cluster, want declarative `kubectl apply` + multi-replica rollout. Uses the official KubeRay `RayService` CRD. |
| **[Modal](modal.md)** | Zero-infra serverless. No cluster, no Ray; pay for actual usage; cold-start costs apply. |

The same `ModelSpec` runs on all three — no code changes between them.
The deployment substrate is what differs.
