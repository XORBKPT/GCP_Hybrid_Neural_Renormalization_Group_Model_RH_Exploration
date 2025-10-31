You're right, I mentioned it in the README but didn't provide the files. My apologies.

It's best practice to separate these into two distinct files, as the dependencies for your GNN training (Team 1 & 2) are different from the dependencies for your data generation (GCP Batch).

Here are the two `requirements.txt` files for your project.

-----

### 1\. For GNN Training (`main_fullbatch.py` & `main_minibatch.py`)

This file contains all the necessary libraries for both training frameworks. Team 2's `main_minibatch.py` specifically requires the `torch-geometric` libraries, while Team 1's `main_fullbatch.py` primarily needs `torch` and `scipy`. This single file will serve both teams.

**File:** `training_requirements.txt`

```text
# Core PyTorch (CPU/CUDA version agnostic)
# Install this first, following https://pytorch.org/
torch
torchvision
torchaudio

# PyTorch Geometric (for Team 2: main_minibatch.py)
# These MUST be installed via the specific PyG/PyTorch/CUDA wheel
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
Please remind Team 2 (Mini-Batch) that they *cannot* just run `pip install -r requirements.txt`. They must install the PyG packages using the specific command from the PyG website that matches your team's PyTorch and CUDA versions (as noted in the README).

-----

### 2\. For Data Generation (`worker.py`, `controller.py`, `aggregator.py`)

This file is much simpler and is used for the "Map-Reduce" data generation jobs. It's what you will build into your `Dockerfile` for the `worker.py` script and what your `controller.py` and `aggregator.py` scripts will need.

**File:** `datagen_requirements.txt`

```text
# For high-precision zero calculation (worker.py)
mpmath

# For GCS & GCP Batch interaction 
# (worker.py, controller.py, aggregator.py)
google-cloud-storage
google-cloud-batch
```
