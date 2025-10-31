Given the goal, **the mini-batch hyperscalability team (Team 2) is significantly more likely to produce the breakthrough** needed for a sub-exponential complexity prediction.

## Team 2 (Hyperscalability)

By sacrificing the `rg_penalty`, they gain the ability to scale.

* *RH Emergent Properties at Scale:* The RH is about *all* zeros ($N \to \infty$). Any emergent property that allows for a sub-exponential prediction (e.g., a hidden fractal structure, a new scaling law) is almost certainly an **emergent property** that is *only* visible at massive $N$.

* An experiment at $N=10k$ or $N=50k$ is, in the context of infinity, like looking at a tiny speck. Team 1 will be is blind to any emergent, high-$N$ phenomena.
* Team 2 gets a JWST "lens" powerful enough to see them, if they are there.

* Riemann and QFT.
    * *Team 1* is testing a *top-down, human-imposed hypothesis*: "We, believe a QFT-like `rg_penalty` *must* be part of the RH mechanism. Let's force the GNN to learn it."
    * A confirmatory, but rigid belief; approach. What if this assumption is wrong?
      
    * *Team 2* is running a *bottom-up, discovery-driven experiment*: "We *assume nothing* about the *mechanism*, only the *statistical signature* (the RMT priors)." so we free up the GNN.
    * It doesn't have to waste capacity learning an artificial `rg_penalty` loss.
    * It is free to find *any* internal representation or mechanism—even one we have no name for—as long as its output *statistically matches* the quantum chaos of the GUE.

RMT priors (NLL/MMD) *are* a fundamental constraint. They model the "what" (the quantum chaos signature) without being prescriptive about the "how" (the `rg_penalty`). 
Team 2's GNN, fed with 100 million data points, is more likely to discover the *true* mechanism (than Team 1's GNN; being forced to learn *our* best *guess* at the mechanism, on a wee dataset.

## 2. BUT the crucial Role of Team 1 is theoretical purity.

Team 1's work is a **critical control experiment** that de-risks the hunt. Team 1's mission is *not* to find "the final answer" but **validate the sacrifice that Team 2 must make.**

Path of A/B testing:

1.  **Initial Race ($N=10k$):** Both teams run their models on the 10k zero dataset.
2.  **Key Question:** Does Team 1's model (with the `rg_penalty`) *significantly* outperform Team 2's model (without it) on extrapolation and statistical accuracy *at this small scale*?
3.  **Two Outcomes:**
    * **Scenario A (Ideal):** Both models perform *similarly well*. It means the `rg_penalty` offers no significant value / redundant.
    * We now have the full justification to drop it and put all resources into Team 2, losing nothing.
      
    * **Scenario B (Problematic):** Team 1's model works, but Team 2's model (without the `rg_penalty`) fails completely.
    * This means the RMT priors *alone* are not a strong-enough signal at small $N$.
    * The project must then find a *scalable* version of the `rg_penalty` or a new, better loss.

So, Team 1 (Full-Batch on A100) is your **Validation & De-risking** group.
Team 2 (Mini-Batch on L4) is your **Discovery & Scaling** group.

Breakthroughs will come from Team 2, but they can only proceed *after* Team 1 gives them the green light.

-----

## GCP Parallel Data Generation Architecture:

The classic "map-reduce" pattern; *map* the work across thousands of small CPUs and then *reduce* the results into one file.

1.  **Controller (Laptop):** A simple Python script that *dispatches* the work. It generates 100,000 "work items" (e.g., "compute zeros 1-1000", "compute 1001-2000", ...) and submits them to GCP Batch.
   
2.  **GCS Bucket (Data Storage):** A central bucket will store two things:
      * `/parts/`: A folder to receive the 100,000+ small text files from the workers.
      * `/final/`: The destination for the final, aggregated `zeta_zeros_100M.txt` file.
        
3.  **Worker (`worker.py` + Docker):** This is the heart of the operation. It's a simple Python script, based on your `mpmath` code, that is containerized with Docker. It's designed to:
      * Wake up.
      * Read its assigned task (e.g., `START_N=1001`, `END_N=2000`).
      * Compute *only* those 1,000 zeros.
      * Write them to a *unique* file (e.g., `zeros_0000001001.txt`).
      * Upload that file to the GCS `/parts/` folder.
      * Shut down.
        
5.  **GCP Batch ("Conductor"):** Fully-managed, does the orchestration. We tell it "run 100,000 copies of my 'Worker' container" and it handles provisioning the VMs, running the jobs, retrying failures, and scaling down.
   
6.  **Aggregator (`aggregator.py`):** Final, single job that runs *after* all workers are done. Lists all 100,000+ files in `/parts/`, downloads and concatenates them *in the correct numerical order*, and streams the final 100M-line file back to `/final/zeta_zeros_100M.txt`.

-----

## 1\. Worker (`worker.py`)

Parameterized using environment variables:

```python
# worker.py
import os
import sys
import mpmath
from google.cloud import storage

print("--- Starting Zeta Zero Worker ---")

# 1. Get job parameters from environment variables
try:
    START_N = int(os.environ['START_N'])
    END_N = int(os.environ['END_N'])
    BUCKET_NAME = os.environ['BUCKET_NAME']
    
    # Format: "parts/zeros_0000001001.txt" (padded for correct sorting)
    FILE_NAME = f"parts/zeros_{START_N:012d}.txt"
    LOCAL_TMP_FILE = "/tmp/zeros.txt"

except KeyError as e:
    print(f"Error: Missing environment variable: {e}")
    sys.exit(1)

print(f"Task: Compute zeros {START_N} to {END_N}")
print(f"Output: gs://{BUCKET_NAME}/{FILE_NAME}")

# 2. Set mpmath precision
mpmath.mp.dps = 30

# 3. Compute the assigned range of zeros
try:
    with open(LOCAL_TMP_FILE, 'w') as f:
        for n in range(START_N, END_N + 1):
            # Write one zero (as a float string) per line
            z = float(mpmath.im(mpmath.zetazero(n)))
            f.write(f"{z}\n")
            if n % 500 == 0:
                print(f"  ... computed zero {n}")

    print(f"Computation complete. Zeros {START_N}-{END_N} written to {LOCAL_TMP_FILE}")

except Exception as e:
    print(f"Error during mpmath computation: {e}")
    sys.exit(1)

# 4. Upload the result file to GCS
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(FILE_NAME)
    
    blob.upload_from_filename(LOCAL_TMP_FILE)
    
    print(f"Successfully uploaded result to gs://{BUCKET_NAME}/{FILE_NAME}")

except Exception as e:
    print(f"Error uploading to GCS: {e}")
    sys.exit(1)

print("--- Zeta Zero Worker Finished ---")
```

## 2\. Worker Environment (`Dockerfile`)

Packages `worker.py` and its dependencies (`mpmath`, `google-cloud-storage`).

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir mpmath google-cloud-storage

# Copy the worker script into the container
WORKDIR /app
COPY worker.py .

# Set the entrypoint
ENTRYPOINT ["python", "worker.py"]
```

**build and push this (one-time setup):**

```bash
# Enable the services
gcloud services enable batch.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create a Docker repository in Artifact Registry
gcloud artifacts repositories create zeta-workers \
    --repository-format=docker \
    --location=europe-west4 # (Use a region near TUM/Munich)

# Build the container
docker build -t zeta-worker .

# Tag it for Artifact Registry
# Make sure to replace <YOUR-GCP-PROJECT-ID>
docker tag zeta-worker europe-west4-docker.pkg.dev/<YOUR-GCP-PROJECT-ID>/zeta-workers/zeta-worker:1.0

# Push it
gcloud auth configure-docker europe-west4-docker.pkg.dev
docker push europe-west4-docker.pkg.dev/<YOUR-GCP-PROJECT-ID>/zeta-workers/zeta-worker:1.0
```

-----

## 3\. Controller (`controller.py`)

Run on your local machine, it generates all the jobs and submits them to GCP Batch.

```python
# controller.py
import os
import sys

# --- CONFIGURATION ---
TOTAL_ZEROS = 10_000_000  # <--- Target: 10 million (or 100M)
CHUNK_SIZE = 1_000        # <--- How many zeros per job
PROJECT_ID = "<YOUR-GCP-PROJECT-ID>"
REGION = "europe-west4"
BUCKET_NAME = "<YOUR-GCS-BUCKET-NAME>" # e.g., "tum-rh-data"
WORKER_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/zeta-workers/zeta-worker:1.0"
# Service account for the Batch jobs to run as
# Needs "Storage Object Admin" role on the bucket
SERVICE_ACCOUNT = f"<YOUR-SERVICE-ACCOUNT-EMAIL>" 

def submit_batch_job(start_n, end_n):
    """Submits a single GCP Batch job."""
    
    job_name = f"zeta-gen-{start_n}"
    
    # This is the gcloud CLI command
    command = f"""
    gcloud batch jobs submit {job_name} \
        --project={PROJECT_ID} \
        --location={REGION} \
        --task-spec='{{
            "runnables": [
                {{
                    "container": {{
                        "imageUri": "{WORKER_IMAGE}",
                        "entrypoint": "python",
                        "commands": ["worker.py"]
                    }}
                }}
            ],
            "environment": {{
                "variables": {{
                    "START_N": "{start_n}",
                    "END_N": "{end_n}",
                    "BUCKET_NAME": "{BUCKET_NAME}"
                }}
            }},
            "computeResource": {{
                "cpuMilli": 1000,
                "memoryMib": 2048
            }}
        }}' \
        --service-account="{SERVICE_ACCOUNT}" \
        --no-sync
    """
    
    print(f"Submitting job for range {start_n}-{end_n}...")
    # We use os.system for simplicity
    # For production, use the Python client library
    os.system(command)

# --- Main execution ---
if __name__ == "__main__":
    if "YOUR-GCP-PROJECT-ID" in PROJECT_ID:
        print("Error: Please configure PROJECT_ID and other variables.")
        sys.exit(1)

    num_jobs = (TOTAL_ZEROS + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"--- Zeta Zero Job Dispatcher ---")
    print(f"Total Zeros: {TOTAL_ZEROS}")
    print(f"Chunk Size:  {CHUNK_SIZE}")
    print(f"Total Jobs:  {num_jobs}")
    
    if input("Proceed to submit? (y/n): ").lower() != 'y':
        print("Aborting.")
        sys.exit(0)

    for i in range(num_jobs):
        start_n = i * CHUNK_SIZE + 1
        end_n = min((i + 1) * CHUNK_SIZE, TOTAL_ZEROS)
        
        submit_batch_job(start_n, end_n)
        
    print(f"All {num_jobs} jobs submitted to GCP Batch.")
```

-----

## 4\. Aggregator (`aggregator.py`)

After GCP Batch shows all jobs are complete, run this *once* (from a Vertex AI Workbench notebook).

```python
# aggregator.py
from google.cloud import storage
import io

# --- CONFIGURATION ---
BUCKET_NAME = "<YOUR-GCS-BUCKET-NAME>"
SOURCE_PREFIX = "parts/"
DESTINATION_BLOB = "final/zeta_zeros_100M.txt"

print("--- Starting Zeta Zero Aggregator ---")
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# 1. List all parts. The padding in the name is critical.
all_parts = list(bucket.list_blobs(prefix=SOURCE_PREFIX))

# Filter out any non-data files and sort them lexicographically
# (e.g., "zeros_0000000001.txt", "zeros_0000001001.txt", ...)
data_parts = [blob for blob in all_parts if blob.name.endswith('.txt')]
data_parts.sort(key=lambda b: b.name)

print(f"Found {len(data_parts)} parts to aggregate.")

# 2. Get the destination blob
final_blob = bucket.blob(DESTINATION_BLOB)

# 3. Stream-compose the final file
# This avoids downloading 100M lines to a local disk.
# We download, write to in-memory buffer, and upload in chunks.
print(f"Composing final file at gs://{BUCKET_NAME}/{DESTINATION_BLOB}...")

with final_blob.open("w") as final_file:
    for i, blob in enumerate(data_parts):
        if i % 100 == 0:
            print(f"  ... processing part {i}/{len(data_parts)} ({blob.name})")
        
        # Download part content as string and write to the final blob
        content = blob.download_as_text()
        final_file.write(content)

print(f"Aggregation complete. Final file created.")
```

This architecture provides you with a robust, scalable, and extremely fast method for generating your training data, reserving your main budget for the A100/L4 machines needed for the GNN training.
