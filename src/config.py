"""Shared paths and environment configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = PROJECT_ROOT / "docs" / "evaluation_rubric.md"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_config() -> tuple[str, str]:
    """Load and validate the OpenAI API key and evaluator model from .env."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("EVALUATOR_MODEL", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Set it locally in .env.")
    if not model:
        raise ValueError("EVALUATOR_MODEL is missing. Set it locally in .env.")
    return api_key, model


def model_options(model: str) -> dict:
    # These non-reasoning model families support temperature zero.
    if model.startswith(("gpt-4.1", "gpt-4o")):
        return {"temperature": 0}
    return {}
