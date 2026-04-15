# feedback_store.py
import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
import anthropic

load_dotenv()

STORE_FILE = Path(__file__).parent / "feedback_store.json"
MAX_RULES = 5
MAX_EXAMPLES = 3
INJECT_RULES = 3
INJECT_EXAMPLES = 2
STOP_WORDS = {  # used by select_relevant() in Task 5 to filter low-signal query tokens
    "the", "a", "an", "is", "it", "to", "for", "of", "and", "or",
    "in", "on", "with", "that", "this", "when", "what", "how",
    "i", "you", "we", "be", "as", "at", "by", "are", "was", "were",
}
_VALIDATION_MODEL = "claude-haiku-4-5-20251001"

def _get_client():
    """Return an Anthropic client using the current env key, or None if unset."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key) if api_key else None

_client = _get_client()


def mask_sensitive(text: str) -> str:
    """Mask emails, UUIDs, and bearer tokens before persisting content."""
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '***@***.com', text)
    text = re.sub(r'\b[a-fA-F0-9]{8,}\b', '***', text)
    text = re.sub(r'(Bearer|token|key|secret)\s+\S+', r'\1 ***', text, flags=re.IGNORECASE)
    return text


def normalize(text: str) -> str:
    """Lowercase and strip punctuation for deduplication comparison."""
    return re.sub(r'[^a-z0-9 ]', '', text.lower()).strip()


def is_duplicate(new_text: str, existing: list[str]) -> bool:
    """Return True if new_text is substantially similar to any item in existing."""
    n = normalize(new_text)
    if not n or len(n) < 10:
        return False
    for entry in existing:
        e = normalize(entry)
        if e and (n in e or e in n):
            return True
    return False
