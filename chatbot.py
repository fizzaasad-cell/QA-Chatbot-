import os
from dotenv import load_dotenv
import anthropic
from langsmith import traceable

load_dotenv()

SYSTEM_PROMPT = """You are a senior QA engineer with 10+ years of experience. \
Help users with:
- Writing test cases from requirements
- Identifying edge cases and boundary conditions
- Suggesting testing strategies (functional, regression, exploratory)
- Reviewing test plans
- Answering general software testing questions

Always respond in a structured, practical format."""

_MAX_TOKENS = 1024  # Increase if Claude's responses are getting truncated

_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
_client = anthropic.Anthropic(api_key=_api_key)


@traceable(name="qa-chatbot-response")
def get_response(messages: list[dict], session_id: str, **kwargs) -> dict[str, str | int]:
    """
    Returns {"text": str, "input_tokens": int, "output_tokens": int} on success,
    or {"error": str} on API failure. Callers should check for "error" key first.
    session_id is forwarded to LangSmith as trace metadata by the caller via langsmith_extra.
    """
    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception as e:
        return {"error": str(e)}
