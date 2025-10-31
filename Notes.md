# Differences between the two codebases and recommendations for deploying this on GCP.

-----

## (a) 🔬 Full-Batch vs. Mini-Batch: A Comparison for Your Teams

The choice is **scale-up vs. scale-out** exploration with A/B testing.

| Feature | `main_fullbatch.py` (Team 1) | `main_minibatch.py` (Team 2) |
| :--- | :--- | :--- |
| **Core** | **Full-Graph (Scale-Up)** | **Graph Partitioning (Scale-Out)** |
| **How to do it** | Loads the *entire* $N \times N$ graph into GPU VRAM for *one* massive computation per epoch. | Uses `Cluster-GCN` to partition the graph into $k$ subgraphs. Loads *one subgraph* at a time into VRAM. |
| **Memory (VRAM)** | **Extremely High.** $O(N \cdot |\text{features}| + |\text{edges}|)$. VRAM is the hard bottleneck. | **Extremely Low.** Depends on cluster size ($N/k$), not $N$. |
| **Scalability** | **Limited.** Hits a hard VRAM wall. $N=10k$ is fine. $N=50k$ might work, $N=100k$ is unlikely. | **Near-Infinite.** Can scale to $N = 10^9+$ by simply increasing $k$ (the number of partitions). |
| **Dependencies** | Self-contained (PyTorch, SciPy). | **Requires PyTorch Geometric (PyG)** and its sparse dependencies. |
| **Training** | **Stable & Deterministic.** The gradient is computed from the *entire* dataset at once. | **Stochastic & Fast.** Each epoch is fast, but gradients are "noisier," as they come from subgraphs. |
| **Loss Function** | **Complete.** Can compute all four losses: MSE, GUE-NLL, GUE-MMD, and the **global `rg_penalty`**. | **Incomplete (by necessity).** |
| **Key Trade-Off** | **Pro:** Can compute the global `rg_penalty` loss, which is theoretically important. <br> **Con:** Cannot scale to massive $N$. | **Pro:** Can scale to *any* $N$. <br> **Con:** **Loses the global `rg_penalty` loss.** This is a *research* trade-off. |

### Implications for Your Both Teams

**Team 1 (Full-Batch):**

  * **Mission:** Your mission is to **test theoretical purity**. The `rg_penalty` (scale-invariance) loss is a core part of your original QFT-inspired hypothesis.
  * **Focus:** You must verify if this loss term is *actually* necessary for convergence and accurate extrapolation.
  * **Limitation:** You will be limited to $N \approx 10k-50k$ zeros. Your goal is to find the best possible result *within this high-but-limited regime*.

**Team 2 (Mini-Batch):**

  * **Mission:** Your mission is to **test for hyperscalability**. You are betting that the RMT priors (NLL and MMD) are *sufficient* to enforce the physics, even without the `rg_penalty`.
  * **Focus:** You must push $N$ as high as possible ($N=100k, 1M, \dots$) to see if new statistical patterns or anomalies emerge at heights no one has ever modeled this way.
  * **Challenge:** You must ensure the RMT loss, computed on *contiguous blocks within clusters*, is a stable-enough signal.

The project's success hinges on answering this question: **Is the `rg_penalty` essential, or can the RMT priors alone guide the model at $N=1M$?** Team 1 and Team 2 will answer this.

-----

## (b) ☁️ GCP Recommendations for Scalable Research

Given your connection to TUM/Google and your Q4 budget, you are in a perfect position to use Google's best-in-class ML infrastructure. The goal is *not* to save money, but to get **answers as fast as possible**.

Here is the optimal setup for each team.

### General GCP Setup (For Both Teams)

1.  **Project & Data:**

      * **Service:** Use **Vertex AI**. Do not manually manage VMs (GCE). Vertex AI is Google's managed platform, built for this.
      * **Data Storage:** Create a **Google Cloud Storage (GCS) Bucket**. Store your `zeta_zeros_10k.txt` (and later, your $1M$ zero file) here. Your training jobs will read directly from this bucket.
      * **Code Repository:** Use **Artifact Registry** to store your custom Docker container images. This is the professional, reproducible way.

2.  **Environment (Container):**

      * Start with a **Vertex AI Deep Learning Container** image. Choose a PyTorch 1.13+ (or 2.x) image with CUDA 12.
      * Create a `Dockerfile` that `FROM` this base image and `pip install torch_geometric` (and its sparse dependencies) and any other requirements.
      * Push this custom container to **Artifact Registry**. Both teams will use this as their training environment.

3.  **Experiment Tracking (Crucial):**

      * Use **Vertex AI Experiments**. Your models have many (hyper)parameters (loss weights, $N$, primes). You *must* log every run.
      * Your Python script can use the Google Cloud AI Platform SDK to log metrics (e.g., `val_loss`, `extrapolation_error`, `gue_nll`) for each epoch. This creates a leaderboard of all your experiments, which is essential for your post-docs.

-----

### Optimal Setup for Team 1: `main_fullbatch.py` (The "Beast")

  * **Problem:** This job is **VRAM-bound**. It needs to fit the *entire* graph, features, and intermediate activations for one $N=10k$ (or $N=50k$) graph into a single GPU.
  * **GCP Service:** **Vertex AI Training** (Custom Job) or **Vertex AI Workbench** (for interactive-but-powerful notebook sessions).
  * **Recommended Machine:** **Accelerator-Optimized (A3 or A2)**.
      * **Primary Choice:** `a2-ultragpu-1g`
          * **GPU:** 1x **NVIDIA A100 80GB**
          * **CPU/RAM:** 12 vCPUs, 136GB RAM
      * **Why:** The **80GB of HBM2e VRAM** is the *only* thing that matters. This machine is designed for massive, single-graph models (like large-language models or, in your case, a massive GNN). This is the "scale-up" solution. Your $50k budget is *exactly* for this class of machine.
      * **Pros:**
          * Fastest possible path to an answer for $N \le 50k$.
          * Allows you to test the *true* full-batch model with the `rg_penalty`.
      * **Cons:**
          * Expensive (but you have the budget).
          * Will *never* scale to $N=1M$. It will fail with an "Out-of-Memory" (OOM) error, and your research will stop.

-----

### Optimal Setup for Team 2: `main_minibatch.py` (The "Swarm")

  * **Problem:** This job is **throughput-bound**. It runs *many* small, fast computations (one per cluster). The GPU doesn't need to be huge; it needs to be *fast* at small-to-medium matrix multiplications.
  * **GCP Service:** **Vertex AI Training** (Custom Job).
  * **Recommended Machine:** **General-Purpose (G2)**.
      * **Primary Choice:** `g2-standard-8`
          * **GPU:** 1x **NVIDIA L4** (24GB VRAM)
          * **CPU/RAM:** 8 vCPUs, 32GB RAM
      * **Why:** An A100 80GB would be *wasted* and *bored*. Each mini-batch (e.g., 200 nodes) is tiny and won't saturate its massive tensor cores. The **L4** is a modern, highly-efficient GPU perfect for this kind of "inference-like" training workload. It will iterate through the batches *very* quickly. The 24GB VRAM is more than enough to hold any single cluster.
      * **Pros:**
          * *Infinitely scalable* in your model (just generate more zeros and more clusters).
          * *Much* cheaper per hour, letting you run more experiments.
      * **Cons:**
          * Loses the `rg_penalty` loss, which is a significant research compromise.
          * The *one-time* graph partitioning by `ClusterData` (for $N=1M$) might take a while, but it only has to be done once.

      93E3 BEBC C164 D766
