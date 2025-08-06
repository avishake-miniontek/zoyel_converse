import os
from huggingface_hub import snapshot_download, login

# === CONFIGURATION (hardcoded) ===
HF_TOKEN = "hf_xxxxxxx"
DEST_ROOT = "./models"
MODEL_IDS = [
    "hexgrad/Kokoro-82M",
    "openai/whisper-large-v3",
]
USE_HF_TRANSFER = True      # whether to enable hf-transfer (may require env var)

# === LOGIN & ENV SETUP ===
login(token=HF_TOKEN)

# optionally enable download accelerator
if USE_HF_TRANSFER:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("⚡ hf_transfer enabled")

# create destination root
os.makedirs(DEST_ROOT, exist_ok=True)

# === DOWNLOAD LOOP ===
for repo_id in MODEL_IDS:
    safe_name = repo_id.replace("/", "--")
    local_dir = os.path.join(DEST_ROOT, safe_name)
    print(f"→ Downloading {repo_id} into {local_dir} …")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        use_auth_token=HF_TOKEN,
        resume_download=True
    )
    print(f"✔ Finished {repo_id}")

print("All downloads complete.")
