"""
utils/config.py
───────────────
Central configuration and LLM factory.
All agents import `get_llm()` from here so switching models is a one-liner.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from crewai import LLM

load_dotenv()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_env() -> dict[str, str]:
    """
    Raise early with clear messages if required env vars are missing.
    Returns the validated keys so callers can surface them in the UI.
    """
    required = {
        "GOOGLE_API_KEY": "Google Gemini API key (https://aistudio.google.com/)",
        "SERPER_API_KEY": "Serper.dev API key  (https://serper.dev/)",
    }
    missing = [f"  • {var}  →  {hint}" for var, hint in required.items() if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            "Missing required API keys. Set them in your .env file or Streamlit secrets:\n"
            + "\n".join(missing)
        )
    return {var: os.environ[var] for var in required}


# ── LLM factory ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_llm(model: str | None = None, temperature: float | None = None) -> LLM:
    """
    Return a cached crewai.LLM instance configured for Gemini via LiteLLM.
    LiteLLM requires the provider prefix "gemini/" on the model string.
    """
    resolved_model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    resolved_temp  = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if not resolved_model.startswith("gemini/"):
        resolved_model = f"gemini/{resolved_model}"

    return LLM(
        model=resolved_model,
        temperature=resolved_temp,
        api_key=os.environ["GOOGLE_API_KEY"],
    )

# ── Misc helpers ──────────────────────────────────────────────────────────────

def output_dir() -> str:
    path = os.getenv("OUTPUT_DIR", "output")
    os.makedirs(path, exist_ok=True)
    return path


def max_iter() -> int:
    return int(os.getenv("MAX_ITER", "8"))
