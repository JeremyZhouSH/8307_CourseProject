import os
from datasets import load_dataset

hf_token = os.getenv("HF_TOKEN")
ds = load_dataset("ccdv/pubmed-summarization", "document", token=hf_token)