"""Upload the fine-tuned model to the Hugging Face Hub (model repositories are free).

Run once from the repository root, after `huggingface_hub` login:  python streamlit_app/upload_model.py
"""
from huggingface_hub import HfApi

REPO = "rabiaozers/minilm-abtbuy"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO, repo_type="model", folder_path="models/minilm-abtbuy")
print("uploaded: https://huggingface.co/" + REPO)
