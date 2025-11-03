*The "two forces" analogy is the best way to frame the problem (see math papers):*

Using the 50-machine "swarm," move from simple training to a massive hyperparameter optimization (HPO). 

Running 50 experiments with random parameters is a waste of money; so go full HPO:

**Phased HPO "Swarm" Strategy**:

Swarm can be managed by **Vertex AI Vizier**, Google's "black-box" Bayesian hyperparameter optimization service <80)

1.  Define the *search space* for your HPO (the ranges for LR, loss weights, etc.).
2.  Ask Vizier for a "trial"—a set of parameters to test.
3.  Launch one of the 50 L4 machines to run that trial.
4.  When the trial is done (e.g., after 100 epochs), report the *result* (your metric, e.g., "validation loss" or "extrapolation error") back to Vizier.
5.  Vizier uses Bayesian optimization to intelligently learn from that result and suggest the *next*, more promising set of parameters for the next free machine.

Instead of a a 50-machine "swarm" brute-force grid search into guided search that will find the optimal balance of the two forces.

---

### Phased HPO: Three Main "Knobs"

Divide a 4-week sprint into phases:

#### Phase 1: Lock the Architecture (First ~3-4 Days)

**Goal:** Find a "good enough" graph structure and a stable optimizer.
**Swarm Allocation:** ~10-15 L4 machines.

**Knob 1: Prime Numbers (Graph Structure)**
* **Idea:** The graph needs to be connected enough to pass information, but not so dense that it becomes a "blender" (a random graph where all nodes are similar).
* **Bet:** The graph is a *secondary* parameter. Its main job is to not be "the broken." (Mandalorain Reference)
* **Plan:** Use a *fixed, high learning rate* (e.g., `1e-3`) and *baseline loss weights* (e.g., `mse=1.0, nll=0.02, mmd=0.05`). Run 5-7 experiments on your $N=100M$ dataset for just a few epochs (long enough to see if the loss plummets or explodes).
    * **Trial 1 (Sparse):** `primes = [2, 3, 5]`
    * **Trial 2 (Baseline):** `primes = [2, 3, 5, 7]`
    * **Trial 3 (Richer):** `primes = [2, 3, 5, 7, 11, 13]`
    * **Trial 4 (Dense):** `primes = [2, 3, 5, 7, 11, 13, 17, 19]`
    * **Trial 5 (Wildcard):** `primes = [11, 13, 17, 19]` (Does the *scale* of primes matter, or just the connectivity?)
* **Decision:** Pick the graph that shows the fastest, most stable loss decrease: **Trial 3 (`[2, 3, 5, 7, 11, 13]`)** is the sweet spot.
* **Action:** **Freeze this graph structure** for all 50 machines for the rest of the sprint.

**Knob 2: Learning Rate & Scheduler**
* **Idea:** A fixed LR is bad for a long run. You *must* use a scheduler to decay the learning rate as the model finds the minimum.
* **Bet:** A standard `CosineAnnealingLR` scheduler is more robust than `ReduceLROnPlateau` for GNNs.
* **Plan:** Using the "winning" graph from above, test 3-4 optimizer configs.
    * **Trial 1:** `lr=5e-4` (w/ CosineAnnealingLR)
    * **Trial 2:** `lr=1e-3` (w/ CosineAnnealingLR)
    * **Trial 3:** `lr=5e-5` (w/ CosineAnnealingLR)
* **Decision:** Pick the one that gives the lowest validation loss after ~10 epochs.
* **Action:** **Freeze this optimizer config** for all 50 machines.

#### Phase 2: Find the Physics (Main Sprint: ~3 Weeks)

**Goal:** This is the core research. Find the optimal *balance* between accuracy (MSE) and the two GUE priors (NLL, MMD).
**Swarm Allocation:** All 50 L4 machines, managed by **Vertex AI Vizier**.

**Knob 3: GUE Loss Weights (Critical Knobs!)**
* **Idea:** The relative weighting of the three losses is the secret.
    * `mse_weight`: Forces the model to be *accurate* (low bias).
    * `gue_nll_weight`: A "sharp" loss. Strongly punishes *individual* spacings that are "non-GUE."
    * `gue_mmd_weight`: A "smooth" loss. Gently guides the *entire distribution* of spacings to look like GUE (low variance).
* **Bet:** A balance. MMD is great for the "big picture," but NLL is the "sergeant" that enforces local discipline. These two are key.
* **Plan (Feed this to Vizier):**
    * Fix `mse_weight = 1.0` (the anchor).
    * Define the search space for the other two. **Use a logarithmic scale.**
    * `gue_nll_weight`: `DoubleParameter(min_value=0.001, max_value=1.0, scale=LOG)`
    * `gue_mmd_weight`: `DoubleParameter(min_value=0.01, max_value=10.0, scale=LOG)`
* **Action:**
    1.  Point all 50 L4s at Vertex AI Vizier.
    2.  Each machine asks Vizier for a (nll_weight, mmd_weight) pair.
    3.  Each machine trains for a *fixed number of epochs* (e.g., 50 epochs).
    4.  It reports its final validation loss back to Vizier.
    5.  Wash, rinse, repeat.
* **Result:** After 3 weeks, Vizier will give you a list of the **Top 10 most powerful loss-weighting combinations** discovered by the 50-machine swarm.

#### Phase 3: Championship Run (Final Week)

**Goal:** Take the "winning" parameters from Phase 2 and train them to full convergence.
**Swarm Allocation:**
* **Team 1 (A100):** Run the *winning* HPO parameters on the full-batch $N=50k$ dataset. This is the Team 1 **control experiment**.
* **Team 2 (L4 Swarm):** Take your Top 10 HPO configs. Dedicate 5 L4s to *each* config (`5 * 10 = 50 machines`). Now, instead of 50 epochs, you let them run for **1000 epochs** or until they fully converge.

### Success is:

**Do not use training loss as the metric for Vizier. Bad idea, I know, we usually do, but not here:**

*The goal is extrapolation. A model that overfits to be *perfectly* accurate at $N=100M$ is useless* here's why:

**Success metric for Vizier will logically be one of these:**

1.  **Good (Standard):** `validation_loss` (loss on the held-out validation set).
2.  **Better (The Real Goal):** A custom "Extrapolation Error" metric.
    * In the training loop, *at the end of every epoch*, freeze the model.
    * Generate the *next 100 zeros* (from $N$ to $N+100$).
    * Compare the *predicted* $N+1$ to $N+100$ against the *actual* (precomputed) values.
    * The `MSE` of *only these 100 future zeros* is the "Extrapolation Error."
    * **Report this "Extrapolation Error" to Vizier.**

This forces the 50-machine swarm to optimize for: **Finding the model that best predicts the future. That's all.** 
