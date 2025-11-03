##Dependencies for GNN training (Team 1 & 2) are different from the dependencies for your data generation (GCP Batch).

-----

### 1\. For GNN Training (`main_fullbatch.py` & `main_minibatch.py`)

Necessary libraries for both training frameworks. 

Team 2's `main_minibatch.py` specifically requires the `torch-geometric` libraries

Team 1's `main_fullbatch.py` needs `torch` and `scipy`. This single file list covers both:

```text
# Core PyTorch (CPU/CUDA version agnostic)
# Install this first, see https://pytorch.org/
torch
torchvision
torchaudio

# PyTorch Geometric (for Team 2: main_minibatch.py)
# These MUST be installed via the specific PyG/PyTorch/CUDA wheel
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# This file lists the packages; the install *method* is critical or stuff will break :0(
torch-geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv

# Common libraries for both teams
numpy
scipy           # For sparse matrix construction (main_fullbatch.py)
networkx        # For graph logic (used in earlier drafts, good to have)
mpmath          # For calculating a few test zeros (e.g., n+1 extrapolation)

# GCP Integration (Optional but recommended)
# For logging to Vertex AI Experiments
google-cloud-aiplatform
```

**Note for PyTorch Geometric:**
Team 2 (Mini-Batch) *cannot* just run `pip install -r requirements.txt`. 
They need to install the PyG packages using the command from the PyG website (that matches their PyTorch and CUDA versions)

-----

### 2\. For Data Generation (`worker.py`, `controller.py`, `aggregator.py`)

This file is easy and used for the "Map-Reduce" data generation jobs. 
Build into your `Dockerfile` for the `worker.py` script and its what your `controller.py` and `aggregator.py` need.

```text
# For high-precision zero calculation (worker.py)
mpmath

# For GCS & GCP Batch interaction 
# (worker.py, controller.py, aggregator.py)
google-cloud-storage
google-cloud-batch
```
