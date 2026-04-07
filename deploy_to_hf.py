import os
import sys
from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    print("Error: HF_TOKEN environment variable is not set. Please provide it to deploy.")
    sys.exit(1)

api = HfApi(token=TOKEN)

# Name of your space
USER = api.whoami()["name"]
SPACE_NAME = "bear-rael-openenv"
REPO_ID = f"{USER}/{SPACE_NAME}"

print(f"Deploying BEAR-RAEL OpenEnv to Hugging Face Spaces: {REPO_ID}...")

# 1. Create the repository if it doesn't exist
try:
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    print("✅ Space repository created / identified.")
except Exception as e:
    print(f"❌ Failed to create or access Space: {e}")
    sys.exit(1)

# 2. Upload the application folder
try:
    print("Uploading project files. This might take a moment...")
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        ignore_patterns=[
            ".git", ".git/*", ".gitignore", "venv", "venv/*", 
            "__pycache__", "*/__pycache__/*", "*.pyc", "deploy_to_hf.py", 
            "inference_output.txt"
        ]
    )
    print(f"\n🎉 Deployment successful!")
    print(f"🌐 View your environment here: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"❌ Failed to upload files: {e}")
    sys.exit(1)
