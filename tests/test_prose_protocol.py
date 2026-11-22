"""The prose protocol: what its instructions ask for, what its parser accepts, and what both
Chat Completions surfaces return when a reply is written in it."""

import json
from typing import Any

import orjson
import pytest
from conftest import MODEL, reply
from pydantic import ValidationError

from app.models import ChatCompletionFunctionTool
from app.utils import g_config
from app.utils.config import GeminiClientSettings
from app.utils.helper import (
    PROSE_MAX_CALLS,
    build_tool_prompt,
    extract_prose_tool_calls,
    process_llm_output,
    prose_retry_nudge,
    tool_argument_max_chars,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Weather in Hanoi?"}]}
OFFERED = ChatCompletionFunctionTool.model_validate(WEATHER_TOOL)


@pytest.fixture
def prose(monkeypatch):
    """Turn the prose protocol on in English, as a fresh install would have it configured."""
    monkeypatch.setattr(g_config.gemini, "tool_protocol", "prose")
    monkeypatch.setattr(g_config.gemini, "tool_protocol_language", "en")
    monkeypatch.setattr(g_config.gemini, "tool_retry_attempts", 1)
    return g_config.gemini


def _arguments(call: Any) -> Any:
    return json.loads(call.function.arguments)


# --- what the instructions ask for ---------------------------------------------------------


def test_the_prompt_describes_an_action_list_and_names_no_serialization_format(prose):
    prompt = build_tool_prompt([OFFERED], "auto")

    assert "action: get_weather" in prompt
    assert "city = " in prompt
    for word in ("tool", "Tool", "function", "json", "JSON", "schema", "{", "}", "[ToolCalls]"):
        assert word not in prompt


def test_the_chinese_wording_is_used_when_it_is_configured(prose, monkeypatch):
    monkeypatch.setattr(g_config.gemini, "tool_protocol_language", "zh")

    prompt = build_tool_prompt([OFFERED], "auto")

    assert "执行" in prompt
    assert "action:" not in prompt


def test_the_prompt_lists_every_offered_action_with_its_arguments(prose):
    prompt = build_tool_prompt([OFFERED], "auto")

    assert "get_weather" in prompt
    assert "city" in prompt
    assert "required" in prompt


# --- what the parser accepts --------------------------------------------------------------


def test_an_action_block_becomes_a_call_and_leaves_the_text(prose):
    visible, calls = extract_prose_tool_calls("Let me look.\n\naction: get_weather\ncity = Hanoi\n")

    assert [(call.function.name, _arguments(call)) for call in calls] == [
        ("get_weather", {"city": "Hanoi"})
    ]
    assert visible == "Let me look."


def test_the_chinese_lead_line_is_accepted_too(prose):
    visible, calls = extract_prose_tool_calls("执行：get_weather\ncity = Hanoi\n")

    assert [call.function.name for call in calls] == ["get_weather"]
    assert visible == ""


def test_a_block_inside_a_code_fence_is_quoted_text_not_a_call(prose):
    quoted = "The page said:\n\n```text\naction: get_weather\ncity = Hanoi\n```\n\nThat is all."

    visible, calls = extract_prose_tool_calls(quoted)

    assert calls == []
    assert "action: get_weather" in visible


def test_only_one_action_is_returned_per_reply(prose):
    reply_text = "action: glob\npattern = *\n\naction: get_weather\ncity = Hanoi\n"

    visible, calls = extract_prose_tool_calls(reply_text)

    assert PROSE_MAX_CALLS == 1
    assert len(calls) == 1
    assert calls[0].function.name == "glob"
    assert "action:" not in visible


def test_a_value_written_over_several_lines_keeps_its_newlines(prose):
    reply_text = "action: write_file\npath = notes.md\ncontent =\n    line one\n    line two\n"

    _, calls = extract_prose_tool_calls(reply_text)

    assert _arguments(calls[0]) == {"path": "notes.md", "content": "line one\nline two"}


def test_numbers_booleans_lists_and_objects_keep_their_types(prose):
    reply_text = (
        'action: setup\nlimit = 12\nratio = 0.5\nforce = true\ntags = [1, 2]\noptions = {"a": 1}\n'
    )

    _, calls = extract_prose_tool_calls(reply_text)

    assert _arguments(calls[0]) == {
        "limit": 12,
        "ratio": 0.5,
        "force": True,
        "tags": [1, 2],
        "options": {"a": 1},
    }


def test_the_text_after_the_block_is_kept(prose):
    visible, calls = extract_prose_tool_calls(
        "action: get_weather\ncity = Hanoi\n\nI will wait for the result."
    )

    assert len(calls) == 1
    assert visible == "I will wait for the result."


def test_a_block_larger_than_the_configured_limit_is_dropped(prose):
    oversized = "x" * (tool_argument_max_chars() + 10)

    visible, calls = extract_prose_tool_calls(f"action: get_weather\ncity = {oversized}\n")

    assert calls == []
    assert oversized not in visible


def test_a_plain_answer_is_not_mistaken_for_a_call(prose):
    visible, calls = extract_prose_tool_calls("Hanoi is warm in November, around 25 degrees.")

    assert calls == []
    assert visible == "Hanoi is warm in November, around 25 degrees."


def test_an_action_word_inside_a_sentence_does_not_open_a_block(prose):
    visible, calls = extract_prose_tool_calls("我不会执行：任何危险操作。这个先放着。")

    assert calls == []
    assert visible == "我不会执行：任何危险操作。这个先放着。"


def test_a_colon_separates_the_name_from_the_value_too(prose):
    _, calls = extract_prose_tool_calls("action: get_weather\ncity: Hanoi\n")

    assert _arguments(calls[0]) == {"city": "Hanoi"}


def test_a_value_may_contain_an_equals_sign(prose):
    _, calls = extract_prose_tool_calls("action: get_weather\ncity = a=b\n")

    assert _arguments(calls[0]) == {"city": "a=b"}


# --- which actions may be called ----------------------------------------------------------


def test_a_call_to_an_action_that_was_not_offered_is_dropped(prose):
    reply_text = "action: get_weather\ncity = Hanoi\n"

    _, _visible, calls = process_llm_output(None, reply_text, None, {"get_weather"})
    _, _, dropped = process_llm_output(None, reply_text, None, {"read_file"})

    assert [call.function.name for call in calls] == ["get_weather"]
    assert dropped == []


def test_without_a_list_of_offered_actions_every_call_is_kept(prose):
    _, _, calls = process_llm_output(None, "action: get_weather\ncity = Hanoi\n", None)

    assert [call.function.name for call in calls] == ["get_weather"]


# --- what the routes return ---------------------------------------------------------------


def test_chat_completions_returns_the_block_as_a_tool_call(api, prose):
    api.upstream.replies.append(reply("action: get_weather\ncity = Hanoi\n"))

    response = api.post(
        "/v1/chat/completions", json={**CHAT, "tools": [WEATHER_TOOL], "tool_choice": "required"}
    )

    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert orjson.loads(call["function"]["arguments"]) == {"city": "Hanoi"}
    assert choice["message"]["content"] in (None, "")


def test_streaming_withholds_the_block_and_ends_with_a_tool_call(api, prose):
    api.upstream.replies.append(reply("Checking.\n\n", "action: get_weather\ncity = Hanoi\n"))

    response = api.post(
        "/v1/chat/completions",
        json={**CHAT, "stream": True, "tools": [WEATHER_TOOL], "tool_choice": "required"},
    )

    assert response.status_code == 200, response.text
    payloads = [
        orjson.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert "action:" not in content
    assert content == "Checking."
    names: list[str] = [
        tool_call["function"]["name"]
        for payload in payloads
        if payload.get("choices")
        for tool_call in payload["choices"][0]["delta"].get("tool_calls") or []
    ]
    assert names == ["get_weather"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_a_forced_choice_is_re_asked_once_when_no_action_is_written(api, prose):
    api.upstream.replies.extend(
        [
            reply("I cannot open files on your computer."),
            reply("action: get_weather\ncity = Hanoi\n"),
        ]
    )

    response = api.post(
        "/v1/chat/completions", json={**CHAT, "tools": [WEATHER_TOOL], "tool_choice": "required"}
    )

    assert response.status_code == 200, response.text
    assert len(api.upstream.sent) == 2
    assert prose_retry_nudge() in api.upstream.sent[1].text
    choice = response.json()["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_an_automatic_choice_is_never_re_asked(api, prose):
    api.upstream.replies.append(reply("Hanoi is warm in November."))

    response = api.post("/v1/chat/completions", json={**CHAT, "tools": [WEATHER_TOOL]})

    assert response.status_code == 200, response.text
    assert len(api.upstream.sent) == 1
    choice = response.json()["choices"][0]
    assert choice["message"].get("tool_calls") in (None, [])
    assert choice["message"]["content"] == "Hanoi is warm in November."


def test_the_pascal_protocol_is_untouched_by_the_prose_parser(monkeypatch):
    monkeypatch.setattr(g_config.gemini, "tool_protocol", "pascal")
    tag_call = (
        "[ToolCalls][Call:get_weather][CallParameter:city]Hanoi[/CallParameter][/Call][/ToolCalls]"
    )

    _, visible, calls = process_llm_output(None, tag_call, None, {"get_weather"})

    assert [call.function.name for call in calls] == ["get_weather"]
    assert visible == ""


# --- configuration ------------------------------------------------------------------------


def _fresh_config(**overrides: Any):
    """A config built from the declared defaults rather than from config.yaml."""
    return type(g_config.gemini)(clients=[GeminiClientSettings(id="c")], **overrides)


def test_the_prose_protocol_is_opt_in_and_defaults_to_the_pascal_protocol():
    fresh = _fresh_config()

    assert fresh.tool_protocol == "pascal"
    assert fresh.tool_protocol_language == "zh"
    assert fresh.tool_retry_attempts == 1
    assert fresh.tool_argument_max_chars == 20000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_protocol", "json"),
        ("tool_protocol_language", "fr"),
        ("tool_retry_attempts", 9),
        ("tool_argument_max_chars", 1),
    ],
)
def test_an_unsupported_protocol_setting_is_rejected(field, value):
    with pytest.raises(ValidationError):
        _fresh_config(**{field: value})
