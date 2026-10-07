import json

import httpx
import pytest
import respx

from himsat.alerts.channels import ChannelError, Message, SmsChannel
from himsat.alerts.llm import LLMClient, LLMError
from himsat.config import Settings

MSG = Message(subject="s", text="t", sms="HimSat: उच्च जोखिम", payload={})
URL = "http://llm.local/v1/chat/completions"


def _ok(content: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


@respx.mock
def test_llm_falls_back_through_response_formats_and_remembers():
    seen = []

    def handler(request):
        fmt = json.loads(request.content).get("response_format", {}).get("type")
        seen.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, text="response_format json_schema not supported")
        return _ok({"title_en": "x"})

    respx.post(URL).mock(side_effect=handler)
    c = LLMClient("http://llm.local/v1", "qwen2.5")
    assert c.chat_json("sys", "user", schema={"type": "object"}) == {"title_en": "x"}
    assert seen == ["json_schema", "json_object"]
    c.chat_json("sys", "user", schema={"type": "object"})
    assert seen[-1] == "json_object" and len(seen) == 3  # accepted mode is tried first next time


@respx.mock
def test_llm_qwen3_gets_no_think_and_errors_are_typed():
    route = respx.post(URL).mock(return_value=_ok({"a": 1}))
    LLMClient("http://llm.local/v1", "Qwen3-8B").chat_json("sys", "hello")
    assert json.loads(route.calls[0].request.content)["messages"][1]["content"].endswith("/no_think")
    respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(LLMError):
        LLMClient("http://llm.local/v1", "m").chat_json("sys", "hello")


def test_llm_disabled_by_settings():
    assert LLMClient.from_settings(Settings(llm_backend="none")) is None


@respx.mock
def test_sparrow_sms_sends_unicode_text():
    s = Settings(sms_provider="sparrow", sparrow_token="tok", sparrow_from="HimSat")
    route = respx.post(s.sparrow_url).mock(return_value=httpx.Response(200, json={"response_code": 200}))
    ch = SmsChannel(s)
    assert ch.available and ch.send("9800000000", MSG) == "200"
    body = route.calls[0].request.content.decode()
    assert "to=9800000000" in body and "%E0%A4" in body  # Devanagari, form-encoded
    respx.post(s.sparrow_url).mock(return_value=httpx.Response(403, text="invalid token"))
    with pytest.raises(ChannelError):
        ch.send("9800000000", MSG)


@respx.mock
def test_twilio_sms_and_unconfigured_provider():
    s = Settings(sms_provider="twilio", twilio_account_sid="AC1", twilio_auth_token="t", twilio_from="+1555")
    respx.post("https://api.twilio.com/2010-04-01/Accounts/AC1/Messages.json").mock(
        return_value=httpx.Response(201, json={"sid": "SM1"}))
    assert SmsChannel(s).send("+977980", MSG) == "SM1"
    none = SmsChannel(Settings(sms_provider="sparrow", sparrow_token=None))
    assert not none.available
