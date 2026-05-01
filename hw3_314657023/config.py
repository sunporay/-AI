import os
from dotenv import load_dotenv

NUM_GPUS = 8
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

def ensure_hf_login():
    load_dotenv("api_key.env")
    hf_token = os.getenv("HF")
    if hf_token:
        from huggingface_hub import login
        try:
            login(token=hf_token)
        except Exception as e:
            print(f"[WARN] Hugging Face login skipped: {e}")
