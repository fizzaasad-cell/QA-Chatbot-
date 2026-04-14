import os
import sys

# Disable LangSmith tracing during tests so no network calls are made
os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ.pop("LANGSMITH_TRACING", None)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import chatbot


def test_get_response_returns_text_and_tokens():
    # Replace the module-level Anthropic client with a mock
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Use equivalence partitioning for login fields.")]
    mock_message.usage.input_tokens = 120
    mock_message.usage.output_tokens = 45
    mock_client.messages.create.return_value = mock_message

    original_client = chatbot._client
    chatbot._client = mock_client

    try:
        result = chatbot.get_response(
            messages=[{"role": "user", "content": "Write test cases for login"}],
            session_id="test-session-abc",
        )
    finally:
        chatbot._client = original_client

    assert result["text"] == "Use equivalence partitioning for login fields."
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 45
