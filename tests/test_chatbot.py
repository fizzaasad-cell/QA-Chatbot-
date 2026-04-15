import os
import sys

os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ.pop("LANGSMITH_TRACING", None)
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
import chatbot


def _mock_client(text="Use equivalence partitioning for login fields.", input_tokens=120, output_tokens=45):
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=text)]
    mock_message.usage.input_tokens = input_tokens
    mock_message.usage.output_tokens = output_tokens
    mock_client.messages.create.return_value = mock_message
    return mock_client


def test_get_response_returns_text_and_tokens():
    mock = _mock_client()
    original = chatbot._client
    chatbot._client = mock
    try:
        result = chatbot.get_response(
            messages=[{"role": "user", "content": "Write test cases for login"}],
            session_id="test-session-abc",
        )
    finally:
        chatbot._client = original

    assert result["text"] == "Use equivalence partitioning for login fields."
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 45
    mock.messages.create.assert_called_once_with(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=chatbot.SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Write test cases for login"}],
    )


def test_get_response_injects_feedback_suffix_when_provided():
    mock = _mock_client()
    original = chatbot._client
    chatbot._client = mock
    feedback = {
        "avoid_rules": ["Do not use vague expected results"],
        "few_shot_examples": [],
        "feedback_log": [],
    }
    try:
        result = chatbot.get_response(
            messages=[{"role": "user", "content": "Write test cases for login"}],
            session_id="test-session-abc",
            feedback=feedback,
        )
    finally:
        chatbot._client = original

    assert "error" not in result
    call_system = mock.messages.create.call_args[1]["system"]
    assert "LEARNED FROM USER FEEDBACK" in call_system
    assert "Do not use vague expected results" in call_system
    assert chatbot.SYSTEM_PROMPT in call_system


def test_get_response_does_not_inject_when_feedback_is_none():
    mock = _mock_client()
    original = chatbot._client
    chatbot._client = mock
    try:
        chatbot.get_response(
            messages=[{"role": "user", "content": "test"}],
            session_id="test-session-abc",
        )
    finally:
        chatbot._client = original

    call_system = mock.messages.create.call_args[1]["system"]
    assert call_system == chatbot.SYSTEM_PROMPT


def test_get_response_returns_error_dict_on_api_failure():
    mock = MagicMock()
    mock.messages.create.side_effect = Exception("Connection timed out")
    original = chatbot._client
    chatbot._client = mock
    try:
        result = chatbot.get_response(
            messages=[{"role": "user", "content": "test"}],
            session_id="test-session-abc",
        )
    finally:
        chatbot._client = original

    assert "error" in result
    assert "Connection timed out" in result["error"]
    assert "text" not in result
