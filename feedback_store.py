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


def _log_entry(
    store: dict,
    rating: str,
    raw_comment: str,
    user_msg: str,
    assistant_msg: str,
    validated: bool,
    distilled_rule: str = "",
    session_id: str = "",
) -> None:
    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "raw_comment": raw_comment,
        "validated": validated,
        "user_message": user_msg,
        "assistant_response": assistant_msg[:500],
    }
    if distilled_rule:
        entry["distilled_rule"] = distilled_rule
    store["feedback_log"].append(entry)


def add_positive(
    user_msg: str,
    assistant_msg: str,
    store: dict,
    session_id: str = "",
) -> str:
    """Validate and save a thumbs-up as a few-shot example.

    Returns: 'saved' | 'skipped_validation' | 'skipped_duplicate'.
    Mutates store in-place and persists to disk on every call.
    """
    masked_user = mask_sensitive(user_msg)
    masked_assistant = mask_sensitive(assistant_msg)

    if not validate_positive(masked_assistant):
        _log_entry(store, "up", "", masked_user, masked_assistant, validated=False, session_id=session_id)
        save_store(store)
        return "skipped_validation"

    existing_assistants = [ex["assistant"] for ex in store["few_shot_examples"]]
    if is_duplicate(masked_assistant, existing_assistants):
        _log_entry(store, "up", "", masked_user, masked_assistant, validated=True, session_id=session_id)
        save_store(store)
        return "skipped_duplicate"

    if len(store["few_shot_examples"]) >= MAX_EXAMPLES:
        store["few_shot_examples"].pop(0)

    store["few_shot_examples"].append({
        "user": masked_user,
        "assistant": masked_assistant,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })

    _log_entry(store, "up", "", masked_user, masked_assistant, validated=True, session_id=session_id)
    save_store(store)
    return "saved"


def add_negative(
    comment: str,
    user_msg: str,
    assistant_msg: str,
    store: dict,
    session_id: str = "",
) -> str:
    """Validate, distill, and save a thumbs-down comment as an avoid-rule.

    Returns: 'saved' | 'skipped_validation' | 'skipped_duplicate'.
    Mutates store in-place and persists to disk on every call.
    """
    masked_comment = mask_sensitive(comment) if comment else ""
    masked_user = mask_sensitive(user_msg)
    masked_assistant = mask_sensitive(assistant_msg)

    if not masked_comment or not validate_negative(masked_comment):
        _log_entry(store, "down", masked_comment, masked_user, masked_assistant, validated=False, session_id=session_id)
        save_store(store)
        return "skipped_validation"

    rule = distill_rule(masked_comment)

    if is_duplicate(rule, store["avoid_rules"]):
        _log_entry(store, "down", masked_comment, masked_user, masked_assistant, validated=True, distilled_rule=rule, session_id=session_id)
        save_store(store)
        return "skipped_duplicate"

    if len(store["avoid_rules"]) >= MAX_RULES:
        store["avoid_rules"].pop(0)

    store["avoid_rules"].append(rule)

    _log_entry(store, "down", masked_comment, masked_user, masked_assistant, validated=True, distilled_rule=rule, session_id=session_id)
    save_store(store)
    return "saved"


def delete_rule(index: int, store: dict) -> None:
    """Remove avoid-rule at index and persist."""
    if 0 <= index < len(store["avoid_rules"]):
        store["avoid_rules"].pop(index)
        save_store(store)


def delete_example(index: int, store: dict) -> None:
    """Remove few-shot example at index and persist."""
    if 0 <= index < len(store["few_shot_examples"]):
        store["few_shot_examples"].pop(index)
        save_store(store)


def _keyword_score(text: str, keywords: set[str]) -> int:
    """Count how many keywords appear in the normalized text."""
    normalized = normalize(text)
    return sum(1 for kw in keywords if kw in normalized)


def select_relevant(store: dict, user_query: str) -> dict:
    """Select the most relevant rules and examples for the current user query.

    Scores entries by keyword overlap with the query.
    Falls back to most-recent entries when no keyword match is found.
    Caps output at INJECT_RULES rules and INJECT_EXAMPLES examples.
    """
    words = set(re.sub(r'[^a-z0-9 ]', '', user_query.lower()).split()) - STOP_WORDS
    rules = store.get("avoid_rules", [])
    examples = store.get("few_shot_examples", [])

    if words:
        scored_rules = sorted(rules, key=lambda r: _keyword_score(r, words), reverse=True)
        scored_examples = sorted(
            examples,
            key=lambda ex: _keyword_score(ex.get("user", "") + " " + ex.get("assistant", ""), words),
            reverse=True,
        )
        top_rules = scored_rules[:INJECT_RULES]
        top_examples = scored_examples[:INJECT_EXAMPLES]

        if not any(_keyword_score(r, words) > 0 for r in top_rules):
            top_rules = rules[-INJECT_RULES:]
        if not any(_keyword_score(ex.get("user", "") + " " + ex.get("assistant", ""), words) > 0 for ex in top_examples):
            top_examples = examples[-INJECT_EXAMPLES:]
    else:
        top_rules = rules[-INJECT_RULES:]
        top_examples = examples[-INJECT_EXAMPLES:]

    return {"avoid_rules": top_rules, "few_shot_examples": top_examples}


def build_feedback_suffix(selected: dict) -> str:
    """Build the prompt suffix string from selected rules and examples.

    Returns empty string if nothing to inject.
    """
    rules = selected.get("avoid_rules", [])
    examples = selected.get("few_shot_examples", [])

    if not rules and not examples:
        return ""

    parts = ["--- LEARNED FROM USER FEEDBACK ---"]

    if rules:
        parts.append("\n## Rules to avoid (based on past feedback):")
        for rule in rules:
            parts.append(f"- {rule}")

    if examples:
        parts.append("\n## Examples of ideal responses:")
        for ex in examples:
            parts.append(f"\nUser: {ex['user']}")
            parts.append(f"Assistant: {ex['assistant']}")

    return "\n".join(parts)
