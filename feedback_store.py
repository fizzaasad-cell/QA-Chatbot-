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


def _empty_store() -> dict:
    return {"avoid_rules": [], "few_shot_examples": [], "feedback_log": []}


def load_store() -> dict:
    """Load feedback_store.json. Returns empty structure if missing or corrupt."""
    if not STORE_FILE.exists():
        return _empty_store()
    try:
        return json.loads(STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()


def save_store(store: dict) -> None:
    """Persist the store dict to feedback_store.json."""
    STORE_FILE.write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_negative(comment: str) -> bool:
    """Ask Claude if this comment is a generalizable QA rule. Returns False on any error."""
    if not _client:
        return False
    try:
        resp = _client.messages.create(
            model=_VALIDATION_MODEL,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    "Is this feedback a generalizable improvement rule for a QA assistant? "
                    f"Answer only yes or no: '{comment}'"
                ),
            }],
        )
        return resp.content[0].text.strip().lower().startswith("yes")
    except Exception:
        return False


def validate_positive(response: str) -> bool:
    """Ask Claude if this response is high-quality. Returns False on any error."""
    if not _client:
        return False
    try:
        resp = _client.messages.create(
            model=_VALIDATION_MODEL,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    "Is this a high-quality, accurate QA assistant response? "
                    f"Answer only yes or no: '{response[:500]}'"
                ),
            }],
        )
        return resp.content[0].text.strip().lower().startswith("yes")
    except Exception:
        return False


def distill_rule(comment: str) -> str:
    """Ask Claude to turn a user complaint into a single rule sentence. Falls back to truncation on error."""
    if not _client:
        return comment[:120]
    try:
        resp = _client.messages.create(
            model=_VALIDATION_MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    "You are helping improve a QA assistant. Distill the following user complaint "
                    "into one concise, actionable rule the assistant should follow. "
                    f"Reply with only the rule sentence, no explanation: '{comment}'"
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        return comment[:120]
