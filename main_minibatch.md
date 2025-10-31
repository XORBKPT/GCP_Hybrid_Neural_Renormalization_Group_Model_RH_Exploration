93E3 BEBC C164 D766

The primary challenge in mini-batching is **preserving the loss structure**.

  * `MSE` loss is fine; it's node-wise.
  * `RG` penalty is a *global* property of the model (comparing two full-graph forward passes). This is incompatible with mini-batching and must be removed or reformulated.
  * `GUE` (NLL/MMD) losses are *structured*; they depend on **contiguous spacings** between adjacent zeros (`pred[i+1] - pred[i]`).

A standard `NeighborLoader` (like for GraphSAGE) *will not work* (it samples random neighbors, breaking ordinal structure). So, partition the graph into **contiguous-ish subgraphs**.

 *`ClusterLoader`** from PyTorch Geometric uses graph clustering (like METIS) to partition the graph. As the Primal Manifold graph has strong `(i, i+1)` path edges, the clusters will be highly likely to contain large, contiguous blocks of nodes (e.g., nodes 1000-1200).

We then modify the loss function to find these contiguous blocks *within* the mini-batch (cluster) and compute the RMT statistics *only* on them.

-----

### 1\. Scalable Mini-Batch Code (`main_minibatch.py`)

Needs PyTorch Geometric.

```python
"""
main_minibatch.py

Hybrid Neural RG Model with Cluster-GCN (PyTorch Geometric) for hyperscalability.
This framework is designed to scale to 100k, 1M, or more zeros, where full-batch
training is no longer feasible.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mpmath
import math
import os
from scipy.sparse import coo_matrix

# --- PyTorch Geometric Imports ---
# Install these. See Setup for instructions.
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.loader as pyg_loader
from torch_geometric.utils import from_scipy_sparse_matrix

# --- 0. Data I/O (Same as full-batch) ---
ZERO_FILE = 'zeta_zeros_10k.txt'
NUM_ZEROS = 10000

def generate_and_save_zeros(num, filename):
    if os.path.exists(filename):
        print(f"[Data] File {filename} already exists. Skipping generation.")
        return
    print(f"[Data] Generating {num} zeros and saving to {filename}...")
    mpmath.mp.dps = 30
    try:
        zeros = [float(mpmath.im(mpmath.zetazero(n))) for n in range(1, num + 1)]
        with open(filename, 'w') as f:
            for z in zeros:
                f.write(f"{z}\n")
        print("[Data] Generation complete.")
    except Exception as e:
        print(f"Error during zero generation: {e}")

def load_zeros(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Zero file not found: {filename}.")
    with open(filename, 'r') as f:
        zeros = [float(line.strip()) for line in f]
    return zeros

# --- 1. Scalable Graph Generation (for PyG) ---

def create_sparse_adelic_graph_edges(num_nodes, primes):
    """
    Creates edge_index and edge_weight for PyG,
    based on the sparse O(N) adjacency matrix.
    """
    print("[Graph] Building sparse adelic graph...")
    rows, cols = [], []
    
    # 1. Path edges (i, i+1)
    for i in range(num_nodes - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        
    # 2. Idelic-inspired edges (sparse)
    for p in primes:
        for res in range(p):
            residue_nodes = [i for i in range(num_nodes) if i % p == res]
            for k in range(len(residue_nodes) - 1):
                u, v = residue_nodes[k], residue_nodes[k+1]
                rows.extend([u, v])
                cols.extend([v, u])
                
    # 3. Self-loops (i, i)
    for i in range(num_nodes):
        rows.append(i)
        cols.append(i)
        
    # Build sparse SciPy matrix
    data = np.ones(len(rows))
    A_sparse = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    A_sparse.eliminate_duplicates() # Important
    
    # --- GCN Normalization (Sparse) ---
    D_diag = np.array(A_sparse.sum(axis=1)).flatten()
    D_inv_sqrt_diag = 1.0 / np.sqrt(D_diag + 1e-6)
    D_inv_sqrt_sparse = coo_matrix((D_inv_sqrt_diag, (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)
    A_norm_sparse = D_inv_sqrt_sparse @ A_sparse @ D_inv_sqrt_sparse
    
    # Convert to PyG edge_index and edge_weight
    edge_index, edge_weight = from_scipy_sparse_matrix(A_norm_sparse)
    
    print(f"[Graph] Graph complete. Nodes: {num_nodes}, Edges: {edge_index.shape[1] // 2}")
    return edge_index, edge_weight

# --- 2. PyG-native GNN Model ---

class HybridNeuralRGGNN_PyG(nn.Module):
    """
    PyG-native version of the Hybrid Neural RG Model.
    Uses built-in GCNConv layers for compatibility with PyG loaders.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(HybridNeuralRGGNN_PyG, self).__init__()
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        # GNN part
        self.gcn1 = pyg_nn.GCNConv(hidden_features, hidden_features)
        self.gcn2 = pyg_nn.GCNConv(hidden_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index, edge_weight):
        """
        Standard PyG forward pass.
        This model is "agnostic" to full-batch or mini-batch.
        The loader handles the data feed.
        """
        x = self.relu(self.mlp(x))
        x = self.relu(self.gcn1(x, edge_index, edge_weight=edge_weight))
        x = self.gcn2(x, edge_index, edge_weight=edge_weight)
        return x

# --- 3. Physics-Informed Loss (Mini-Batch Version) ---

# GUE PDF (NLL) and MMD Sampler/Kernel (same as full-batch)
def gue_pdf(s):
    return (32 / (math.pi ** 2)) * (s ** 2) * torch.exp(-4 * (s ** 2) / math.pi)

def sample_wigner_surmise(n_samples, device):
    samples = []
    max_pdf = 0.6
    while len(samples) < n_samples:
        s = torch.rand(1, device=device) * 3.0
        y = torch.rand(1, device=device) * max_pdf
        if y <= gue_pdf(s):
            samples.append(s)
    return torch.cat(samples)

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist_sq = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2
    return torch.exp(-beta * dist_sq).mean()

def compute_gue_mmd_loss(pred_spacings, n_samples=500, device='cpu'):
    if pred_spacings.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    true_spacings = sample_wigner_surmise(n_samples, device=device)
    k_xx = gaussian_kernel(pred_spacings, pred_spacings)
    k_yy = gaussian_kernel(true_spacings, true_spacings)
    k_xy = gaussian_kernel(pred_spacings, true_spacings)
    return k_xx + k_yy - 2 * k_xy

def rg_loss_minibatch(pred, target, original_indices, weights, device):
    """
    Mini-batch compatible loss function.
    
    !! CRITICAL: This version omits the global 'rg_penalty' !!
    It is computationally incompatible with mini-batching.
    
    It computes RMT statistics *only* on contiguous blocks of zeros
    found within the mini-batch (cluster).
    """
    
    # L1: MSE (always computable)
    mse = nn.MSELoss()(pred, target)
    
    # L2 & L3: RMT Priors (NLL & MMD)
    gue_nll = torch.tensor(0.0, device=device)
    gue_mmd = torch.tensor(0.0, device=device)
    
    # We must find contiguous (i, i+1) spacings
    # 1. Sort the batch by the original node indices
    sorted_idx = torch.sort(original_indices).indices
    sorted_preds = pred[sorted_idx]
    sorted_original_indices = original_indices[sorted_idx]
    
    # 2. Find where the original indices are contiguous (diff == 1)
    index_diffs = sorted_original_indices[1:] - sorted_original_indices[:-1]
    contiguous_mask = (index_diffs == 1)
    
    if contiguous_mask.sum() > 1:
        # 3. Calculate spacings ONLY for these contiguous predictions
        contiguous_spacings = (sorted_preds[1:] - sorted_preds[:-1])[contiguous_mask]
        
        # We must have positive spacings for GUE
        contiguous_spacings = contiguous_spacings[contiguous_spacings > 0]
        
        if contiguous_spacings.numel() > 1:
            mean_spacing = torch.mean(contiguous_spacings)
            unfolded = contiguous_spacings / (mean_spacing + 1e-10)
            unfolded = unfolded[unfolded > 0]
            
            if unfolded.numel() > 0:
                # GUE NLL
                pdf_values = gue_pdf(unfolded)
                pdf_values = torch.clamp(pdf_values, min=1e-10)
                gue_nll = -torch.mean(torch.log(pdf_values))
                
                # GUE MMD
                gue_mmd = compute_gue_mmd_loss(unfolded, n_samples=200, device=device)

    # Combine losses
    total_loss = (weights['mse'] * mse +
                  weights['gue_nll'] * gue_nll +
                  weights['gue_mmd'] * gue_mmd)
    
    return total_loss, (mse, gue_nll, gue_mmd)


# --- 4. Training and Evaluation Functions ---

def train(model, train_loader, optimizer, loss_weights, device):
    model.train()
    total_loss = 0
    total_mse = 0
    total_nll = 0
    total_mmd = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass on the subgraph (cluster)
        pred = model(batch.x, batch.edge_index, batch.edge_weight)
        
        # Get predictions and targets for nodes in this cluster
        pred_nodes = pred[batch.train_mask]
        target_nodes = batch.y[batch.train_mask]
        # We need the *original* indices to find spacings
        original_indices = batch.n_id[batch.train_mask] 
        
        loss, (mse, nll, mmd) = rg_loss_minibatch(
            pred_nodes, target_nodes, original_indices, loss_weights, device
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_nll += nll.item()
        total_mmd += mmd.item()
        
    num_batches = len(train_loader)
    return (total_loss / num_batches, 
            total_mse / num_batches, 
            total_nll / num_batches, 
            total_mmd / num_batches)

@torch.no_grad()
def evaluate(model, data, val_loader, loss_weights, device):
    """
    Evaluates on the validation set.
    For validation, we still use the RMT loss to get a comparable metric.
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_nll = 0
    total_mmd = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        
        pred = model(batch.x, batch.edge_index, batch.edge_weight)
        
        pred_nodes = pred[batch.val_mask]
        target_nodes = batch.y[batch.val_mask]
        original_indices = batch.n_id[batch.val_mask]
        
        loss, (mse, nll, mmd) = rg_loss_minibatch(
            pred_nodes, target_nodes, original_indices, loss_weights, device
        )
        
        total_loss += loss.item()
        total_mse += mse.item()
        total_nll += nll.item()
        total_mmd += mmd.item()
        
    num_batches = len(val_loader)
    return (total_loss / num_batches, 
            total_mse / num_batches, 
            total_nll / num_batches, 
            total_mmd / num_batches)

# --- 5. Main Execution ---

if __name__ == '__main__':
    
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Setup] Using device: {device}")
    
    # !! Run this once to create the 'zeta_zeros_10k.txt' file !!
    # generate_and_save_zeros(NUM_ZEROS, ZERO_FILE)
    # import sys; sys.exit()

    try:
        zeros = load_zeros(ZERO_FILE)
        if len(zeros) != NUM_ZEROS:
            print(f"Warning: Loaded {len(zeros)} zeros, expected {NUM_ZEROS}.")
    except FileNotFoundError as e:
        print(e)
        print("Please run the script once with 'generate_and_save_zeros' uncommented.")
        import sys; sys.exit()
        
    primes = [2, 3, 5, 7, 11] # Primes for adelic structure
    
    # --- 2. Create PyG Data Object ---
    edge_index, edge_weight = create_sparse_adelic_graph_edges(NUM_ZEROS, primes)
    
    features = np.log(np.arange(1, NUM_ZEROS + 1) + 1e-6).reshape(-1, 1)
    node_features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(zeros, dtype=torch.float32).unsqueeze(1)
    
    # Split: 90% train (9k), 10% val (1k)
    train_split = int(NUM_ZEROS * 0.9)
    train_mask = torch.zeros(NUM_ZEROS, dtype=torch.bool)
    train_mask[:train_split] = True
    val_mask = torch.zeros(NUM_ZEROS, dtype=torch.bool)
    val_mask[train_split:] = True
    
    data = pyg_data.Data(
        x=node_features, 
        y=targets, 
        edge_index=edge_index, 
        edge_weight=edge_weight,
        train_mask=train_mask,
        val_mask=val_mask,
        n_id=torch.arange(NUM_ZEROS) # Store original indices
    )
    
    # --- 3. Create ClusterLoader ---
    # Partition the 10k graph into 100 clusters (avg 100 nodes each)
    # This is the core of the mini-batching strategy
    print("[Cluster] Partitioning graph for Cluster-GCN...")
    cluster_data = pyg_loader.ClusterData(data, num_parts=100, save_dir='clusters')
    # batch_size=1 means "one cluster per batch"
    train_loader = pyg_loader.ClusterLoader(cluster_data, batch_size=1, shuffle=True)
    val_loader = pyg_loader.ClusterLoader(cluster_data, batch_size=1, shuffle=False)
    print("[Cluster] Partitioning complete.")

    # --- 4. Model, Optimizer, Loss ---
    model = HybridNeuralRGGNN_PyG(1, 128, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 400 # May need fewer epochs as each epoch sees all data
    
    loss_weights = {
        'mse': 1.0,
        'gue_nll': 0.02,
        'gue_mmd': 0.05
    }

    # --- 5. Training Loop ---
    print(f"[Train] Starting mini-batch training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss, train_mse, train_nll, train_mmd = train(
            model, train_loader, optimizer, loss_weights, device
        )
        
        if epoch % 20 == 0:
            val_loss, val_mse, val_nll, val_mmd = evaluate(
                model, data, val_loader, loss_weights, device
            )
            print(f"--- Epoch {epoch} ---")
            print(f"  Train: Loss={train_loss:.4f} (MSE={train_mse:.4f}, NLL={train_nll:.4f}, MMD={train_mmd:.4f})")
            print(f"  Val:   Loss={val_loss:.4f} (MSE={val_mse:.4f}, NLL={val_nll:.4f}, MMD={val_mmd:.4f})")

    print("[Train] Training complete.")

    # --- 6. Extrapolation Test ---
    # Note: For inference, it's best to switch back to a full-batch
    # forward pass, as we are not memory-constrained (no backprop).
    print("\n--- Extrapolation Test (using full-graph inference) ---")
    
    try:
        actual_next = float(mpmath.im(mpmath.zetazero(NUM_ZEROS + 1)))
        print(f"Actual {NUM_ZEROS + 1}-st zero: {actual_next:.4f}")
    except Exception:
        actual_next = -1
        
    num_extended = NUM_ZEROS + 1
    
    # Extend graph and features for one new node
    ext_edge_index, ext_edge_weight = create_sparse_adelic_graph_edges(num_extended, primes)
    ext_features = np.log(np.arange(1, num_extended + 1) + 1e-6).reshape(-1, 1)
    ext_node_features = torch.tensor(ext_features, dtype=torch.float32)

    ext_edge_index = ext_edge_index.to(device)
    ext_edge_weight = ext_edge_weight.to(device)
    ext_node_features = ext_node_features.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_ext = model(ext_node_features, ext_edge_index, ext_edge_weight)
        predicted_next = pred_ext[-1].item()

    print(f"Predicted {NUM_ZEROS + 1}-st zero: {predicted_next:.4f} (actual {actual_next:.4f})")

```

