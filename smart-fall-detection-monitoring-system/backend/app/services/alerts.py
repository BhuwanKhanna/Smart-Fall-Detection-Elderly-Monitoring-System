from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

import requests

from backend.app.core.config import Settings


class AlertDispatcher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dispatch(self, event: dict[str, Any]) -> list[str]:
        channels: list[str] = []
        if self.settings.alert_email_enabled:
            channels.append(self._send_email(event))
        if self.settings.alert_sms_enabled:
            channels.append(self._send_sms(event))
        return [channel for channel in channels if channel]

    def _send_email(self, event: dict[str, Any]) -> str:
        subject = f"[Safety Alert] {event['event_type'].replace('_', ' ').title()}"
        body = self._format_message(event)

        if not all(
            [
                self.settings.smtp_username,
                self.settings.smtp_password,
                self.settings.smtp_sender,
                self.settings.emergency_contact_email,
            ]
        ):
            print(f"[SIMULATED EMAIL] To={self.settings.emergency_contact_email} Subject={subject}\n{body}")
            return "simulated-email"

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.settings.smtp_sender
        message["To"] = self.settings.emergency_contact_email
        message.set_content(body)

        try:
            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.settings.smtp_username, self.settings.smtp_password)
                server.send_message(message)
            return "email"
        except Exception as exc:
            print(f"[EMAIL ERROR] {exc}")
            return "simulated-email"

    def _send_sms(self, event: dict[str, Any]) -> str:
        if not all(
            [
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
                self.settings.twilio_from_number,
                self.settings.emergency_contact_phone,
            ]
        ):
            print(f"[SIMULATED SMS] To={self.settings.emergency_contact_phone} {self._format_message(event)}")
            return "simulated-sms"

        try:
            response = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.settings.twilio_account_sid}/Messages.json",
                auth=(self.settings.twilio_account_sid, self.settings.twilio_auth_token),
                data={
                    "From": self.settings.twilio_from_number,
                    "To": self.settings.emergency_contact_phone,
                    "Body": self._format_message(event),
                },
                timeout=10,
            )
            response.raise_for_status()
            return "sms"
        except Exception as exc:
            print(f"[SMS ERROR] {exc}")
            return "simulated-sms"

    @staticmethod
    def _format_message(event: dict[str, Any]) -> str:
        return (
            f"Emergency monitoring detected {event['event_type'].replace('_', ' ')} "
            f"with confidence {event['confidence']:.2f}. "
            f"Severity: {event['severity']}. "
            f"Timestamp: {event['created_at']}."
        )
