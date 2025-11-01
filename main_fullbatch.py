"""
93E3 BEBC C164 D766

main_fullbatch.py

Hybrid Neural RG Model with Full-Batch Training.

Designed for one (powerful) GCP instance and does the sparse graph, 
the MMD "Hamiltonian simulation" loss (pre quantum compute)
and RG-flow penalty.

Sparse matrices handle N=10,000+ nodes in a single batch, 
suitable for high-memory GPU instances (e.g., GCP A100).

Includes physics-informed losses (check math XORBKPT repo for the theory):
1. MSE (Accuracy)
2. RG-Flow Penalty (Scale Invariance)
3. GUE-NLL (RMT Prior)
4. GUE-MMD (Hamiltonian Simulation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import mpmath
import math
import os
import sys
import argparse
from scipy.sparse import coo_matrix  # For building sparse matrices efficiently

# --- 0. Configuration ---
ZERO_FILE = 'zeta_zeros_10k.txt'
NUM_ZEROS = 10000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Data Generation (Run Once) ---

def generate_and_save_zeros(num, filename):
    """Generates and saves the first num zeta zeros to a file."""
    if os.path.exists(filename):
        print(f"[Data] File {filename} already exists. Skipping generation.")
        return
    
    print(f"[Data] Generating {num} zeros and saving to {filename}...")
    mpmath.mp.dps = 30 # Set precision
    try:
        zeros = [float(mpmath.im(mpmath.zetazero(n))) for n in range(1, num + 1)]
        with open(filename, 'w') as f:
            for z in zeros:
                f.write(f"{z}\n")
        print(f"[Data] Generation complete. Saved {len(zeros)} zeros.")
    except Exception as e:
        print(f"Error during zero generation: {e}")
        print("This can happen if mpmath's database is exceeded. Try a smaller number.")
        
# --- 2. Data Loading ---

def load_zeros(filename):
    """Loads zeros from a precomputed file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Zero file not found: {filename}. Run --generate_zeros first.")
    
    with open(filename, 'r') as f:
        zeros = [float(line.strip()) for line in f]
    print(f"[Data] Loaded {len(zeros)} zeros from {filename}.")
    return zeros

# --- 3. Scalable Graph Generation (Sparse Adelic) ---

def create_sparse_adelic_graph(num_nodes, primes):
    """
    Creates a sparse adjacency matrix (A_norm) in PyTorch sparse format.
    This fixes the O(N^2) density issue from the original code.
    """
    print("[Graph] Building sparse adelic graph...")
    
    rows = []
    cols = []
    
    # 1. Path edges (i, i+1)
    for i in range(num_nodes - 1):
        rows.append(i)
        cols.append(i + 1)
        rows.append(i + 1)
        cols.append(i)
        
    # 2. Idelic-inspired edges (sparse)
    for p in primes:
        for res in range(p):
            residue_nodes = [i for i in range(num_nodes) if i % p == res]
            for k in range(len(residue_nodes) - 1):
                u, v = residue_nodes[k], residue_nodes[k+1]
                rows.append(u)
                cols.append(v)
                rows.append(v)
                cols.append(u)
                
    # 3. Add self-loops (i, i)
    for i in range(num_nodes):
        rows.append(i)
        cols.append(i)
        
    # Build sparse SciPy matrix
    data = np.ones(len(rows))
    A_sparse = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    A_sparse.eliminate_duplicates() # Remove any duplicate edges
    
    # --- GCN Normalization (Sparse) ---
    D_diag = np.array(A_sparse.sum(axis=1)).flatten()
    D_inv_sqrt_diag = 1.0 / np.sqrt(D_diag + 1e-6)
    D_inv_sqrt_sparse = coo_matrix((D_inv_sqrt_diag, (range(num_nodes), range(num_nodes))), shape=(num_nodes, num_nodes), dtype=np.float32)
    
    # A_norm = D^{-1/2} A D^{-1/2}
    A_norm_sparse = D_inv_sqrt_sparse @ A_sparse @ D_inv_sqrt_sparse
    
    # Convert to PyTorch sparse tensor
    coo = A_norm_sparse.tocoo()
    values = torch.tensor(coo.data, dtype=torch.float32)
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    
    A_norm_torch_sparse = torch.sparse_coo_tensor(indices, values, torch.Size(coo.shape))
    
    print(f"[Graph] Graph complete. Nodes: {num_nodes}, Edges: {A_sparse.nnz // 2}")
    return A_norm_torch_sparse.to(DEVICE)

# --- 4. Sparse GNN Model ---

class GCNLayer(nn.Module):
    """Sparse GCN Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        nn.init.xavier_uniform_(self.weight) # Better initialization
    
    def forward(self, H, A_norm_sparse):
        # A_norm_sparse is (N, N), H is (N, in_features)
        support = torch.sparse.mm(A_norm_sparse, H) # A_norm * H
        output = support @ self.weight # (A_norm * H) * W
        return output

class HybridNeuralRGGNN(nn.Module):
    """Full Hybrid Neural RG Model"""
    def __init__(self, in_features, hidden_features, out_features):
        super(HybridNeuralRGGNN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )
        self.gcn1 = GCNLayer(hidden_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, H, A_norm_sparse):
        # H is the node feature matrix (N, in_features)
        # A_norm_sparse is the normalized adjacency matrix (N, N)
        H = self.relu(self.mlp(H))
        H = self.relu(self.gcn1(H, A_norm_sparse))
        H = self.gcn2(H, A_norm_sparse)
        return H

# --- 5. Physics-Informed Loss (Full Suite) ---

def gue_pdf(s):
    """GUE PDF (Wigner Surmise, beta=2)."""
    return (32 / (math.pi ** 2)) * (s ** 2) * torch.exp(-4 * (s ** 2) / math.pi)

def sample_wigner_surmise(n_samples, device):
    """Samples from GUE PDF. This is our 'Hamiltonian Simulation'."""
    samples = []
    max_pdf = 0.6  # Approximate max
    while len(samples) < n_samples:
        s = torch.rand(1, device=device) * 3.0  # GUE PDF negligible > 3
        y = torch.rand(1, device=device) * max_pdf
        if y <= gue_pdf(s):
            samples.append(s)
    # Concatenate all tensors in the list
    if not samples:
        return torch.tensor([], device=device)
    return torch.cat(samples)

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian kernel for MMD."""
    beta = 1.0 / (2.0 * sigma**2)
    dist_sq = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2
    return torch.exp(-beta * dist_sq).mean()

def compute_gue_mmd_loss(pred_spacings, n_samples=500, device='cpu'):
    """MMD loss comparing predicted spacings to GUE simulation."""
    if pred_spacings.numel() == 0:
        return torch.tensor(0.0, device=device)
    
    true_spacings = sample_wigner_surmise(n_samples, device=device)
    if true_spacings.numel() == 0:
        return torch.tensor(0.0, device=device)
        
    k_xx = gaussian_kernel(pred_spacings, pred_spacings)
    k_yy = gaussian_kernel(true_spacings, true_spacings)
    k_xy = gaussian_kernel(pred_spacings, true_spacings)
    
    mmd_loss = k_xx + k_yy - 2 * k_xy
    return mmd_loss

def rg_loss_full(pred, target, mask, model, H, A_norm_sparse, weights):
    """
    Full physics-informed loss for the full-batch model.
    Includes MSE, RG-Penalty, GUE-NLL, and GUE-MMD.
    """
    masked_pred = pred[mask]
    masked_target = target[mask]
    
    # L1: MSE (Accuracy)
    mse = nn.MSELoss()(masked_pred, masked_target)
    
    # L2: RG-Flow Penalty (Scale Invariance)
    # This term is only possible in a full-batch model
    with torch.no_grad(): # Don't track grads for the 'scaled' input
        scale_factor = 2.0
        # Log-scale features to simulate dilation
        scaled_H = torch.log(H * scale_factor + 1e-6)
        scaled_pred = model(scaled_H, A_norm_sparse)
    
    # Beta-inspired penalty: Should be ~ pred / scale_factor if invariant
    rg_penalty = torch.mean((scaled_pred[mask] / scale_factor - masked_pred)**2)
    
    # L3 & L4: RMT Priors (NLL & MMD)
    gue_nll = torch.tensor(0.0, device=DEVICE)
    gue_mmd = torch.tensor(0.0, device=DEVICE)
    
    if masked_pred.size(0) > 1:
        # Calculate spacings *only* on the training data
        pred_diffs = masked_pred[1:] - masked_pred[:-1]
        
        # Filter out any non-positive spacings
        pred_diffs = pred_diffs[pred_diffs > 0]
        
        if pred_diffs.numel() > 1:
            mean_spacing = torch.mean(pred_diffs)
            unfolded = pred_diffs / (mean_spacing + 1e-10) # Unfold
            unfolded = unfolded[unfolded > 0]
            
            if unfolded.numel() > 0:
                # L3: GUE NLL
                pdf_values = gue_pdf(unfolded)
                pdf_values = torch.clamp(pdf_values, min=1e-10)
                gue_nll = -torch.mean(torch.log(pdf_values))
                
                # L4: GUE MMD ("Hamiltonian Simulation")
                # Sample 500 or numel, whichever is smaller
                n_samples = min(500, unfolded.numel())
                gue_mmd = compute_gue_mmd_loss(unfolded, n_samples=n_samples, device=DEVICE)

    # Combine losses
    total_loss = (weights['mse'] * mse +
                  weights['rg'] * rg_penalty +
                  weights['gue_nll'] * gue_nll +
                  weights['gue_mmd'] * gue_mmd)
    
    return total_loss, (mse, rg_penalty, gue_nll, gue_mmd)

# --- 6. Main Execution ---

def main():
    
    # --- 1. Setup ---
    print(f"[Setup] Using device: {DEVICE}")
    
    # Load data
    try:
        zeros = load_zeros(ZERO_FILE)
    except FileNotFoundError as e:
        print(e)
        print(f"Please run: python {sys.argv[0]} --generate_zeros {NUM_ZEROS}")
        sys.exit(1)
        
    if len(zeros) != NUM_ZEROS:
        print(f"Warning: Loaded {len(zeros)} zeros, expected {NUM_ZEROS}.")
        
    primes = [2, 3, 5, 7, 11, 13] # Richer adelic structure
    
    # --- 2. Create Graph and Features ---
    A_norm_sparse = create_sparse_adelic_graph(NUM_ZEROS, primes)
    
    features = np.log(np.arange(1, NUM_ZEROS + 1) + 1e-6).reshape(-1, 1)
    node_features = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(zeros, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    
    # Split: 90% train, 10% val
    train_split = int(NUM_ZEROS * 0.9)
    train_mask = torch.zeros(NUM_ZEROS, dtype=torch.bool).to(DEVICE)
    train_mask[:train_split] = True
    val_mask = torch.zeros(NUM_ZEROS, dtype=torch.bool).to(DEVICE)
    val_mask[train_split:] = True
    
    # --- 3. Model, Optimizer, Loss ---
    model = HybridNeuralRGGNN(1, 128, 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 2000 # Full-batch may need more epochs for fine-tuning
    
    loss_weights = {
        'mse': 1.0,
        'rg': 0.1,      # Your original RG-flow weight
        'gue_nll': 0.02,  # Your optimized GUE-NLL weight
        'gue_mmd': 0.05   # New "Hamiltonian" loss weight (tune this)
    }

    print("[Train] Starting full-batch training on 10,000 zeros...")
    
    # --- 4. Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Full-batch forward pass
        pred = model(node_features, A_norm_sparse)
        
        loss, components = rg_loss_full(
            pred, targets, train_mask, model, node_features, A_norm_sparse, loss_weights
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Get full validation loss
                val_pred = model(node_features, A_norm_sparse)
                val_loss_mse = nn.MSELoss()(val_pred[val_mask], targets[val_mask])
            
            mse, rg, nll, mmd = components
            print(f"--- Epoch {epoch} ---")
            print(f"  Train Loss: {loss.item():.4f} (MSE={mse.item():.4f}, RG={rg.item():.4f}, NLL={nll.item():.4f}, MMD={mmd.item():.4f})")
            print(f"  Val MSE:    {val_loss_mse.item():.4f}")

    print("[Train] Training complete.")

    # --- 5. Extrapolation Test (Predict n=10001) ---
    print("\n--- Extrapolation Test ---")
    
    try:
        mpmath.mp.dps = 30
        actual_next = float(mpmath.im(mpmath.zetazero(NUM_ZEROS + 1)))
        print(f"Actual {NUM_ZEROS + 1}-st zero: {actual_next:.4f}")
    except Exception as e:
        print(f"Could not compute {NUM_ZEROS + 1}-st zero: {e}")
        actual_next = -1
        
    num_extended = NUM_ZEROS + 1
    
    # Re-build graph for N+1 nodes (fast)
    A_norm_ext_sparse = create_sparse_adelic_graph(num_extended, primes).to(DEVICE)
    
    features_ext = np.log(np.arange(1, num_extended + 1) + 1e-6).reshape(-1, 1)
    node_features_ext = torch.tensor(features_ext, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        pred_ext = model(node_features_ext, A_norm_ext_sparse)
        predicted_next = pred_ext[-1].item()

    print(f"Predicted {NUM_ZEROS + 1}-st zero: {predicted_next:.4f} (actual {actual_next:.4f})")
    print(f"Extrapolation Error: {abs(predicted_next - actual_next):.4f}")


if __name__ == '__main__':
    # --- Argument Parser for Data Generation ---
    parser = argparse.ArgumentParser(description="Full-Batch Neural RG Model")
    parser.add_argument(
        '--generate_zeros', 
        type=int, 
        metavar='N',
        help='Generates the first N zeta zeros and saves to file, then exits.'
    )
    
    args = parser.parse_args()
    
    if args.generate_zeros:
        print(f"--- Data Generation Mode ---")
        generate_and_save_zeros(args.generate_zeros, ZERO_FILE.replace('10k', f'{args.generate_zeros}'))
        print("Data generation complete. Exiting.")
    else:
        # Set config based on file
        if os.path.exists(ZERO_FILE):
             NUM_ZEROS = len(load_zeros(ZERO_FILE))
             print(f"[Setup] Found data file, setting NUM_ZEROS={NUM_ZEROS}")
        else:
             print(f"[Setup] No data file found. Using default NUM_ZEROS={NUM_ZEROS}")
             print(f"Please run: python {sys.argv[0]} --generate_zeros {NUM_ZEROS}")
             sys.exit(1)
             
        main()
