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

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@traceable(name="qa-chatbot-response")
def get_response(messages: list, session_id: str) -> dict:
    response = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return {
        "text": response.content[0].text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
