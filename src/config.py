import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("TEAMIFIED_OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment")

PDF_PATH = "data\\PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5
