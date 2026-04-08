import os
import sys
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    print("Error: HF_TOKEN environment variable is not set.")
    sys.exit(1)

api = HfApi(token=TOKEN)
USER = api.whoami()["name"]
SPACE_NAME = "bear-rael-openenv"
REPO_ID = f"{USER}/{SPACE_NAME}"

print(f"Deploying to: {REPO_ID}")

try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=[".git*", "venv*", "__pycache__*", "deploy_to_hf.py"]
    )
    print(f"🎉 Success! https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)
