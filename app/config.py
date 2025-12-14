import os
from dotenv import load_dotenv

# Root of the project (same logic as your test script, but adjusted for /app directory)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from app/
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Load .env from project root, overriding any existing env vars
load_dotenv(ENV_PATH, override=True)

# Uploads folder (PDF + extracted images)
BASE_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Folder for generated labeled images (legacy, if ever used)
IMAGE_OUTPUT_DIR = os.path.join(BASE_UPLOAD_DIR, "generated")

# Folder containing static educational organ images
# e.g. static/organs/heart.jpg, lungs.jpg, etc.
ORGAN_IMAGE_DIR = os.path.join(BASE_DIR, "static", "organs")

os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(ORGAN_IMAGE_DIR, exist_ok=True)

# ====== Azure OpenAI environment configuration ======
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or ""
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or ""
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or ""
AZURE_OPENAI_VISION_DEPLOYMENT = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT") or ""
