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


def test_load_store_returns_empty_structure_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_store, "STORE_FILE", tmp_path / "nonexistent.json")
    store = feedback_store.load_store()
    assert store["avoid_rules"] == []
    assert store["few_shot_examples"] == []
    assert store["feedback_log"] == []


def test_load_store_returns_empty_structure_on_corrupt_file(tmp_path, monkeypatch):
    f = tmp_path / "bad.json"
    f.write_text("not valid json")
    monkeypatch.setattr(feedback_store, "STORE_FILE", f)
    store = feedback_store.load_store()
    assert store["avoid_rules"] == []


def test_save_and_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback_store, "STORE_FILE", tmp_path / "store.json")
    store = {"avoid_rules": ["rule1"], "few_shot_examples": [], "feedback_log": []}
    feedback_store.save_store(store)
    loaded = feedback_store.load_store()
    assert loaded["avoid_rules"] == ["rule1"]


from unittest.mock import MagicMock, patch


def test_validate_negative_returns_true_on_yes():
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="yes")]
    with patch.object(feedback_store._client, "messages") as m:
        m.create.return_value = mock_resp
        assert feedback_store.validate_negative("missed boundary values") is True


def test_validate_negative_returns_false_on_no():
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="no")]
    with patch.object(feedback_store._client, "messages") as m:
        m.create.return_value = mock_resp
        assert feedback_store.validate_negative("I don't like it") is False


def test_validate_negative_returns_false_on_api_error():
    with patch.object(feedback_store._client, "messages") as m:
        m.create.side_effect = Exception("timeout")
        assert feedback_store.validate_negative("some comment") is False


def test_validate_positive_returns_true_on_yes():
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Yes")]
    with patch.object(feedback_store._client, "messages") as m:
        m.create.return_value = mock_resp
        assert feedback_store.validate_positive("TC_LOGIN_001 - Title: Verify that...") is True


def test_validate_positive_returns_false_on_api_error():
    with patch.object(feedback_store._client, "messages") as m:
        m.create.side_effect = Exception("network error")
        assert feedback_store.validate_positive("some response") is False


def test_distill_rule_returns_cleaned_sentence():
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text="Do not skip boundary values for numeric fields.")]
    with patch.object(feedback_store._client, "messages") as m:
        m.create.return_value = mock_resp
        result = feedback_store.distill_rule("you missed boundary values on the password field")
    assert result == "Do not skip boundary values for numeric fields."


def test_distill_rule_falls_back_to_truncated_comment_on_error():
    with patch.object(feedback_store._client, "messages") as m:
        m.create.side_effect = Exception("timeout")
        result = feedback_store.distill_rule("x" * 200)
    assert len(result) <= 120
