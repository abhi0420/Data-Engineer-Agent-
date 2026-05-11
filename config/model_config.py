# config/model_config.py
import os
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = ["openai", "anthropic", "google"]

def get_llm(temperature: float = 0.2, max_tokens: int = 1000):
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

    else:
        raise ValueError(f"Unsupported provider '{provider}'. Choose from: {SUPPORTED_PROVIDERS}")