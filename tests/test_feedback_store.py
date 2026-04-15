import sys, os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import feedback_store


def test_mask_sensitive_masks_email():
    result = feedback_store.mask_sensitive("Contact me at user@example.com please")
    assert "user@example.com" not in result
    assert "***@***.com" in result


def test_mask_sensitive_masks_uuid():
    result = feedback_store.mask_sensitive("Token is abc12345def67890")
    assert "abc12345def67890" not in result
    assert "***" in result


def test_mask_sensitive_masks_bearer():
    result = feedback_store.mask_sensitive("Authorization: Bearer eyJhbGciOiJSUzI1")
    assert "eyJhbGciOiJSUzI1" not in result


def test_mask_sensitive_leaves_normal_text():
    result = feedback_store.mask_sensitive("Write test cases for login")
    assert result == "Write test cases for login"


def test_normalize_lowercases_and_strips_punctuation():
    assert feedback_store.normalize("Hello, World!") == "hello world"


def test_normalize_handles_empty():
    assert feedback_store.normalize("") == ""


def test_is_duplicate_detects_substring():
    existing = ["Do not skip boundary values for numeric fields"]
    assert feedback_store.is_duplicate("Do not skip boundary values", existing) is True


def test_is_duplicate_detects_superset():
    existing = ["skip boundary"]
    assert feedback_store.is_duplicate("Do not skip boundary values for numeric fields", existing) is True


def test_is_duplicate_returns_false_for_distinct():
    existing = ["Do not use vague expected results"]
    assert feedback_store.is_duplicate("Always include preconditions", existing) is False


def test_is_duplicate_empty_list():
    assert feedback_store.is_duplicate("any rule", []) is False


def test_is_duplicate_returns_false_for_very_short_new_text():
    existing = ["Do not skip boundary values for numeric fields"]
    assert feedback_store.is_duplicate("skip", existing) is False
