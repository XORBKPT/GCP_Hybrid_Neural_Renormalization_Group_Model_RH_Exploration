**This repository contains the research framework and code for a hybrid machine learning model aimed at verifying the Riemann Hypothesis at high heights.**
*Intended for post-doctoral researchers in number theory and quantum field theory on weekends.*


The RH is a statement about infinity. Any true mechanism allowing for sub-exponential prediction is likely an emergent property that *only* manifests at massive scale. Team 2's GNN, fed with 100M+ data points and constrained only by the *statistical signature* of quantum chaos (the RMT priors), is free to discover a "true" internal mechanism that we (as human researchers), may not even have the language for yet.

Emergent Properties at Scale: So the RH *is* a statement about all zeros N ==> Infinity. This much we know. Therefore any property that allows for a sub-exponential prediction (e.g., a hidden fractal structure, a new scaling law) is almost certainly an emergent property that is only visible at massive heights. Team 1 is permanently blind to these emergent, high-N phenomena. Team 2 is the only one of the two that has a JWST lens powerful enough to see them, if there.

The model operationalizes a novel dynamical framework by merging Renormalization Group (RG) flows with Graph Neural Networks (GNNs). The GNN models the "Primal Manifold," a profinite space encoding primes as topological defects, while physics-informed losses enforce theoretical priors from Random Matrix Theory (RMT) and QFT scale-invariance.

## Overview

The core hypothesis is that the zeros of the Riemann zeta function can be predicted by a GNN learning the RG flow dynamics on a graph representing the Primal Manifold. The model's components are:

1.  **Primal Manifold Graph:** A sparse graph where nodes are the ordinal indices of the zeros.
2.  **Hybrid GNN-MLP:** A GNN (using `GCNConv` layers) processes the graph structure, while an MLP backbone processes the node features.
3.  *Physics-Informed Losses:*
    * **MSE:** Standard regression loss for positional accuracy.
    * **RMT (GUE) Prior:** We enforce the Montgomery Conjecture by matching the statistics of the *predicted unfolded spacings* to the Gaussian Unitary Ensemble (GUE). This is done via two complementary losses:
        * **GUE-NLL:** A Negative Log-Likelihood loss against the GUE PDF (Wigner Surmise).
        * **GUE-MMD:** A Maximum Mean Discrepancy loss, which compares the *distribution* of predicted spacings to a "simulated Hamiltonian" (samples from the GUE PDF).
    * **RG-Flow Penalty:** (Full-batch only) A loss term that enforces scale invariance, a key property of the RG fixed point at the critical line.

## Full-Batch vs. Mini-Batch | Deploying on GCP:

Choice: **scale-up vs. scale-out** - do both - an exploration with A/B testing.

| Feature | `main_fullbatch.py` (Team 1) | `main_minibatch.py` (Team 2) |
| :--- | :--- | :--- |
| **Core** | **Full-Graph (Scale-Up)** | **Graph Partitioning (Scale-Out)** |
| **How to do it** | Loads the *entire* $N \times N$ graph into GPU VRAM for *one* massive computation per epoch. | Uses `Cluster-GCN` to partition the graph into $k$ subgraphs. Loads *one subgraph* at a time into VRAM. |
| **Memory (VRAM)** | **Extremely High.** VRAM is the hard bottleneck. | **Extremely Low.** Depends on cluster size (N/k), not N. |
| **Scalability** | **Limited.** Hits a hard VRAM wall. N=10k is fine. N=50k might work, N=100k unlikely. | **Near-Infinite.** Can scale to N = 10^9 or more by increasing k (number of partitions). |
| **Dependencies** | Self-contained (PyTorch, SciPy). | **Requires PyTorch Geometric (PyG)** and its sparse dependencies. |
| **Training** | **Stable & Deterministic.** The gradient is computed from the *entire* dataset at once. | **Stochastic & Fast.** Each epoch is fast, but gradients are "noisier," as they come from subgraphs. |
| **Loss Function** | **Complete.** Can compute all four losses: MSE, GUE-NLL, GUE-MMD, and the **global `rg_penalty`**. | **Incomplete (by necessity).** |
| **Key Trade-Off** | **Pro:** computes the global `rg_penalty` loss, which is theoretically important. <br> **Con:** Cannot scale to massive $N$. | **Pro:** Can scale to *any* $N$. <br> **Con:** **Loses the global `rg_penalty` loss.** A research trade-off, this is fine. |

### Trade-Off: `rg_penalty`

The `rg_penalty` loss from main_fullbatch code:
`rg_penalty = torch.mean((scaled_pred / scale_factor - pred)**2)`
needs *two full-graph forward passes* to compute; all good in full-batch.

In the mini-batch, its computationally infeasible. We *cannot* check a *global* scale-invariance property using a small subgraph.

So, `main_minibatch.py` **removes this loss term**. It relies totally on the RMT (NLL and MMD) losses to act as the physics-informed regularizer. This is a necessary research trade-off for scalability. The `main_fullbatch.py` script *retains* this loss, as it uses the "compute version" of the Hamiltonian simulation (compute, non-QC, included)

---

## Setup

Use a `conda` environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create Conda Environment:**
    ```bash
    conda create -n neural-rg python=3.10
    conda activate neural-rg
    ```

3.  **Install PyTorch:**
    (Visit [pytorch.org](https://pytorch.org/) for the command specific to your CUDA version.)
    ```bash
    # Example for CUDA 12.1
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4.  **Install PyTorch Geometric (PyG):**
    This is required for the mini-batch framework. PyG's installation is tied to your PyTorch/CUDA version.
    ```bash
    # Example for PyTorch 2.1 / CUDA 12.1
    pip install torch_geometric
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f [https://data.pyg.org/whl/torch-2.1.0+cu121.html](https://data.pyg.org/whl/torch-2.1.0+cu121.html)
    ```

5.  **Install other dependencies:**
    ```bash
    pip install numpy networkx mpmath scipy
    ```

---

## Run

### Step 1: Generate Dataset

The script will generate the first 10,000 (or more) zeros and save them to `zeta_zeros_10k.txt`.

We use the `main_fullbatch.py` script for this, as it contains the generation helper.

*For larger data sets use generate_and_save_zeros.md*

```bash
# Generate the first 10,000 zeros
python main_fullbatch.py --generate_zeros=10000
```
*Note: Generating 10k zeros can take several minutes. Generating 100k+ can take hours.*

### Step 2 (Option A): Run the Full-Batch Model

This is the baseline, good for $ N=10,000 $ on a system with a powerful GPU (e.g., > 32GB VRAM).

```bash
python main_fullbatch.py
```
This script will:
1.  Load all 10,000 zeros.
2.  Build the full $ 10000 \times 10000 $ sparse adjacency matrix in memory.
3.  Perform full-batch gradient descent.
4.  Print training progress and final extrapolation.

### Step 2 (Option B): Run the Mini-Batch Model

This is the scalable framework, recommended for N 10,000 or for systems with limited VRAM.

```bash
python main_minibatch.py
```
This script will:
1.  Load all 10,000 zeros.
2.  Build the graph and partition it into 100 clusters (subgraphs), saving them to a new `clusters/` directory. (Only on the first run).
3.  Iterate through the *clusters* (1 cluster = 1 mini-batch).
4.  Train the model, computing RMT losses only on contiguous spacings found *within* each cluster.
5.  Print training progress and final extrapolation.

## Code:

```
.
├── main_fullbatch.py   # Full-batch model (Baseline, N<=50k)
├── main_minibatch.py   # Mini-batch model (Scalable, N~1M+)
├── README.md           # This file
|
├── zeta_zeros_10k.txt  # Data file (generated)
└── clusters/           # Directory for PyG clusters (generated)
```

## Next Steps

* **Scaling $N$:** Use the `main_minibatch.py` framework to train on N=10^5, 10^6 ....to check for emergent statistical anomalies at extreme heights.
* **Full Adelic Graph:** Enhance the `create_sparse_adelic_graph` function to model the full adelic space, incorporate the "archimedean" component or more complex p-adic topologies.
* **Hamiltonian Simulation:** The GUE-MMD loss is our "compute version" of a Hamiltonian simulation. Replace this statistical prior with a true quantum algorithm (e.g., VQE, QPE) on a QC to find the spectrum of a candidate Hilbert-Pólya operator.

93E3 BEBC C164 D766
