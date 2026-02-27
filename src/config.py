import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

# PDF_PATH = "YOUR LOCAL PATH"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 2
