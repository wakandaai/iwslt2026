from huggingface_hub import hf_hub_download
import os

repo_id = "WakandaAI/Aura-1B"
output_dir = "/ocean/projects/cis250145p/tanghang/iwslt2026/checkpoints/aura_1b"
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading checkpoint from {repo_id}...")

files = ["model.pt", "tokenizer.json"]
for filename in files:
    print(f"Downloading {filename}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
    )
    print(f"  → {path}")

print("\nDone.")
print(f"Downloaded to: {path}")

