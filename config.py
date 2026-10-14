import os
from dotenv import load_dotenv

# 从 .env 读取密钥（.env 不入库，避免 key 泄露）
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("请在 .env 中设置 OPENAI_API_KEY（DeepSeek 的 API Key）")
