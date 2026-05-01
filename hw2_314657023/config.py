import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv("api_key.env")
hf_token = os.getenv("HF")
if hf_token:
    login(token=hf_token)

NUM_GPUS = 8
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"