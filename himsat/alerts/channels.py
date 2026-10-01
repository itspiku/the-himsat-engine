"""Delivery channels: SMS (Sparrow SMS – Nepal, Twilio), e-mail (SMTP), signed webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

import httpx

from himsat.config import Settings, get_settings

log = logging.getLogger(__name__)


class ChannelError(RuntimeError):
    pass


@dataclass
class Message:
    subject: str
    text: str
    sms: str
    payload: dict


class SmsChannel:
    name = "sms"

    def __init__(self, s: Settings | None = None):
        self.s = s or get_settings()

    @property
    def available(self) -> bool:
        if self.s.sms_provider == "sparrow":
            return bool(self.s.sparrow_token and self.s.sparrow_from)
        if self.s.sms_provider == "twilio":
            return bool(self.s.twilio_account_sid and self.s.twilio_auth_token and self.s.twilio_from)
        return False

    def send(self, to: str, msg: Message) -> str:
        if self.s.sms_provider == "sparrow":
            # Sparrow SMS (Nepal): https://docs.sparrowsms.com — Unicode text is supported
            r = httpx.post(self.s.sparrow_url, data={"token": self.s.sparrow_token, "from": self.s.sparrow_from,
                                                     "to": to, "text": msg.sms}, timeout=30)
            if r.status_code >= 300:
                raise ChannelError(f"Sparrow SMS {r.status_code}: {r.text[:200]}")
            return str(r.json().get("response_code", r.status_code)) if r.headers.get(
                "content-type", "").startswith("application/json") else str(r.status_code)
        if self.s.sms_provider == "twilio":
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.s.twilio_account_sid}/Messages.json"
            r = httpx.post(url, data={"To": to, "From": self.s.twilio_from, "Body": msg.sms},
                           auth=(self.s.twilio_account_sid or "", self.s.twilio_auth_token or ""), timeout=30)
            if r.status_code >= 300:
                raise ChannelError(f"Twilio {r.status_code}: {r.text[:200]}")
            return r.json().get("sid", "")
        raise ChannelError("no SMS provider configured")


class EmailChannel:
    name = "email"

    def __init__(self, s: Settings | None = None):
        self.s = s or get_settings()

    @property
    def available(self) -> bool:
        return bool(self.s.smtp_host)

    def send(self, to: str, msg: Message) -> str:
        em = EmailMessage()
        em["Subject"] = msg.subject
        em["From"] = f"{self.s.alert_sender_name} <{self.s.smtp_from}>"
        em["To"] = to
        em.set_content(msg.text)
        try:
            with smtplib.SMTP(self.s.smtp_host or "", self.s.smtp_port, timeout=30) as smtp:
                if self.s.smtp_starttls:
                    smtp.starttls()
                if self.s.smtp_user:
                    smtp.login(self.s.smtp_user, self.s.smtp_password or "")
                smtp.send_message(em)
        except (smtplib.SMTPException, OSError) as e:
            raise ChannelError(f"SMTP: {e}") from e
        return em.get("Message-ID", "") or "sent"


def sign_payload(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class WebhookChannel:
    name = "webhook"

    def __init__(self, s: Settings | None = None):
        self.s = s or get_settings()

    @property
    def available(self) -> bool:
        return True

    def send(self, to: str, msg: Message) -> str:
        body = json.dumps(msg.payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8", "User-Agent": "HimSat-Engine/1.0"}
        if self.s.webhook_secret:
            headers["X-HimSat-Signature"] = sign_payload(body, self.s.webhook_secret)
        try:
            r = httpx.post(to, content=body, headers=headers, timeout=20)
        except httpx.HTTPError as e:
            raise ChannelError(f"webhook: {e}") from e
        if r.status_code >= 300:
            raise ChannelError(f"webhook {r.status_code}: {r.text[:200]}")
        return str(r.status_code)


def get_channels(s: Settings | None = None) -> dict[str, object]:
    s = s or get_settings()
    return {"sms": SmsChannel(s), "email": EmailChannel(s), "webhook": WebhookChannel(s)}
