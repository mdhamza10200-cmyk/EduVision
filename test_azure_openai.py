import os
from dotenv import load_dotenv
from openai import AzureOpenAI

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(ENV_PATH, override=True)

print("Endpoint from .env:", os.getenv("AZURE_OPENAI_ENDPOINT"))

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

resp = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
    messages=[{"role": "user", "content": "Say hello from Azure OpenAI."}],
)

print("Response:", resp.choices[0].message.content)
