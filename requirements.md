The dependencies for GNN training (Team 1 & 2) are different from the dependencies for data generation (GCP Batch).

-----

### 1\. For GNN Training (`main_fullbatch.py` & `main_minibatch.py`)

Libraries for both training frameworks. Team 2's `main_minibatch.py` specifically requires the `torch-geometric` libraries, while Team 1's `main_fullbatch.py` needs `torch` and `scipy`. This file serves both.

```text
# Core PyTorch (CPU/CUDA version agnostic)
# Install this first, following https://pytorch.org/
torch
torchvision
torchaudio

# PyTorch Geometric (for Team 2: main_minibatch.py)
# These must be installed via the specific PyG/PyTorch/CUDA wheel
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# This file lists the packages; the install *method* is critical.
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
Team 2 (Mini-Batch) *cannot* just run `pip install -r requirements.txt`. They must install the PyG packages using the specific command from the PyG website that matches your team's PyTorch and CUDA versions (see README).

-----

### 2\. Data Generation (`worker.py`, `controller.py`, `aggregator.py`)

Used for the "Map-Reduce" data generation jobs. It's what you'll build into your `Dockerfile` for the `worker.py` script and what your `controller.py` and `aggregator.py` scripts will need.

**File:** `datagen_requirements.txt`

```text
# For high-precision zero calculation (worker.py)
mpmath

# For GCS & GCP Batch interaction 
# (worker.py, controller.py, aggregator.py)
google-cloud-storage
google-cloud-batch
```
93E3 BEBC C164 D766
