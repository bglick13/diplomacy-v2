# Cloud Infrastructure Research

Research into alternative cloud providers for our ML training workload.

---

## Current Requirements

| Component | Resources | Current (Modal) |
|-----------|-----------|-----------------|
| Inference | ~12 A100s | Autoscaled, serverless |
| Training | 1 H100 | Single GPU container |
| Rollouts | ~100 CPU workers | `modal.Function.map()` |

---

## The Network Overhead Problem

Current architecture has network boundaries between all components:

```
[CPU Rollout Workers] <--network--> [A100 Inference] <--network--> [H100 Trainer]
       Modal                              Modal                        Modal
```

Each inference call crosses Modal's network boundary. With 100 rollout workers making sequential calls, latency adds up.

---

## Options Comparison

### Option 1: Modal (Current)

| Pros | Cons |
|------|------|
| Zero ops overhead | Network latency between services |
| Pay-per-use | No co-location possible |
| Fast iteration | A100 ~$3.50/hr (expensive vs alternatives) |

**Cost estimate**: 12 A100s × $3.50 × 24hr = **~$1,000/day**

---

### Option 2: Ray Cluster on CoreWeave/GKE

Run everything on one Kubernetes cluster with Ray:

```
[Ray Cluster on Kubernetes]
├── Head Node (CPU) - orchestration
├── CPU Workers (100 pods) - rollouts
├── GPU Workers (12× A100) - vLLM inference via Ray Serve
└── GPU Worker (1× H100) - training
```

| Pros | Cons |
|------|------|
| **Same-cluster networking** (~1ms latency) | Kubernetes ops overhead |
| vLLM + Ray Serve native integration | Need to manage autoscaling |
| Reserved pricing (up to 60% off) | Minimum commitments |
| Free egress within CoreWeave | |

**CoreWeave Cost**:
- 12× A100 80GB: ~$2.21/hr × 12 = $26.52/hr
- 1× H100: ~$4.25/hr
- 100 CPU cores: ~$5/hr (estimate)
- **Total: ~$36/hr = ~$864/day** (on-demand)
- **With 60% reserved discount: ~$350/day**

---

### Option 3: Lambda Labs Reserved Cluster

| Pros | Cons |
|------|------|
| $2.99/hr H100 (cheapest) | **Minimum 64 GPUs** for reserved |
| Pre-configured ML stack | Overkill for our scale |
| InfiniBand interconnect | |

**Not ideal** - minimum 64 GPUs is way more than needed.

---

### Option 4: RunPod Instant Clusters

| Pros | Cons |
|------|------|
| A100: ~$1.33/hr (community) | Less enterprise support |
| No minimum commitment | "Community" = shared hardware |
| Fast spin-up | |

**Cost estimate**: 12× A100 × $1.33 = ~$16/hr = **~$384/day**

---

## GPU Pricing Summary (2025)

### H100 Pricing

| Provider | $/GPU/hour | Notes |
|----------|------------|-------|
| Lambda Labs | $2.99 | Cheapest, but 64 GPU minimum for reserved |
| RunPod (Community) | $1.99 | Shared hardware |
| RunPod (Secure) | $2.39 | Dedicated node |
| CoreWeave | $4.25-6.15 | Includes InfiniBand |
| Modal | ~$4.50 | Serverless premium |

### A100 Pricing

| Provider | $/GPU/hour | Notes |
|----------|------------|-------|
| RunPod (Community) | $1.33 | Shared hardware |
| RunPod (Secure) | $1.90 | Dedicated node |
| CoreWeave | $2.21 | Before CPU/RAM charges |
| Modal | ~$3.50 | Serverless premium |

---

## Ray + vLLM Architecture Benefits

### Co-located Inference

With Ray Serve + vLLM:
- Rollout workers and inference on same cluster
- ~10-100x lower latency for inference calls
- Ray handles heterogeneous CPU/GPU scheduling

### vLLM Distributed Features

- Tensor parallelism across GPUs
- Pipeline parallelism across nodes
- Native Ray integration for multi-node
- Dynamic batching via Ray Serve

### Example Configuration

```python
# Ray Serve with vLLM backend
from ray import serve
from vllm import AsyncLLMEngine

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 4, "max_replicas": 12}
)
class VLLMDeployment:
    def __init__(self):
        self.engine = AsyncLLMEngine.from_engine_args(...)
```

---

## Recommendation

### Short-term: Optimize Modal
- Autoscaling fix (target_inputs=8) should help
- Network overhead may be acceptable for batch sizes
- Lowest operational complexity

### Medium-term: Ray on CoreWeave/GKE

Best option for our scale:
1. Deploy Ray cluster on GKE or CoreWeave
2. Run vLLM with Ray Serve for inference
3. Run rollout workers as Ray tasks on CPU nodes
4. Keep trainer on same cluster

Benefits:
- ~10-100x lower latency for inference calls
- ~40-60% cost reduction with reserved instances
- Ray handles heterogeneous CPU/GPU scheduling natively

---

## Sources

- [CoreWeave GPU Pricing](https://www.coreweave.com/pricing)
- [Lambda Labs Pricing](https://lambda.ai/pricing)
- [RunPod Pricing](https://www.runpod.io/pricing)
- [vLLM Distributed Serving](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html)
- [Ray on GKE Quickstart](https://cloud.google.com/kubernetes-engine/docs/add-on/ray-on-gke/quickstarts/ray-gpu-cluster)
- [Ray Serve with vLLM](https://www.anyscale.com/blog/deepseek-vllm-ray-google-kubernetes)
- [H100 Pricing Comparison (Nov 2025)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
