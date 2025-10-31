Yes. This is a perfect use case for a "fan-out" parallel architecture.

The `generate_and_save_zeros` function is what's known as an **embarrassingly parallel** task: computing zero $n=90,000,000$ has zero dependence on zero $n=10$. Your current script is **serial**, doing them one by one. On a single core, this would take months or years.

We will adapt this by turning your serial `for` loop into **100,000+ parallel jobs** running simultaneously on **GCP Batch**. Instead of one machine taking months, 10,000 machines will each work for a few minutes. This will turn this "impossible" data generation task into something that completes in an afternoon.

This architecture is extremely cost-effective (likely a few hundred dollars, not your full $50k budget) and is the standard way to handle "big compute" data generation at Google.

-----

## 🚀 The Parallel Data Generation Architecture

This is the "map-reduce" pattern. We'll *map* the work across thousands of small CPUs and then *reduce* the results into one file.

1.  **Controller (Your Laptop):** A simple Python script that *dispatches* the work. It generates 100,000 "work items" (e.g., "compute zeros 1-1000", "compute 1001-2000", ...) and submits them to GCP Batch.
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
4.  **GCP Batch (The "Foreman"):** This fully-managed service does all the orchestration. You tell it "run 100,000 copies of my 'Worker' container" and it handles provisioning the VMs, running the jobs, retrying failures, and scaling down.
5.  **Aggregator (`aggregator.py`):** A final, single job that runs *after* all workers are done. It lists all 100,000+ files in `/parts/`, downloads and concatenates them *in the correct numerical order*, and streams the final 100M-line file back to `/final/zeta_zeros_100M.txt`.

-----

## 1\. The Worker (`worker.py`)

This script is the "work" to be done. It's parameterized using environment variables.

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

## 2\. The Worker Environment (`Dockerfile`)

We need to package `worker.py` and its dependencies (`mpmath`, `google-cloud-storage`).

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

**To build and push this (one-time setup):**

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

## 3\. The Controller (`controller.py`)

This is the script you run on your local machine. It generates all the jobs and submits them to GCP Batch.

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

## 4\. The Aggregator (`aggregator.py`)

After GCP Batch shows all jobs are complete, run this *once* (e.g., from a powerful Vertex AI Workbench notebook).

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
