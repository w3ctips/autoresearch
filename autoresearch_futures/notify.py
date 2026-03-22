"""
Signal notification module.
Pushes trading signals via WeChat, Telegram, and Email.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from autoresearch_futures.config import NotifyConfig

SIGNAL_COOLDOWN_MINUTES = 30
_last_signals: Dict[str, datetime] = {}


@dataclass
class SignalEvent:
    """Container for a trading signal event."""
    symbol: str
    direction: int
    timestamp: datetime
    price: float
    confidence: float
    sources: List[str]
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None


def format_signal_message(signal: SignalEvent) -> str:
    """Format signal as human-readable message."""
    direction_text = "做多" if signal.direction == 1 else "做空"
    sources_text = "\n".join(f"- {s}" for s in signal.sources)

    msg = f"""【期货信号提醒】
品种: {signal.symbol}
方向: {direction_text}
时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}
价格: {signal.price:.2f}
置信度: {signal.confidence:.0%}

理论依据:
{sources_text}
"""
    if signal.suggested_stop:
        msg += f"\n建议止损: {signal.suggested_stop:.2f}"
    if signal.suggested_target:
        msg += f"\n建议止盈: {signal.suggested_target:.2f}"

    return msg


def push_wechat(message: str, webhook: str) -> bool:
    """Push message to WeChat (企业微信机器人)."""
    try:
        response = requests.post(
            webhook,
            json={"msgtype": "text", "text": {"content": message}},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"WeChat push failed: {e}")
        return False


def push_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """Push message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram push failed: {e}")
        return False


def push_email(message: str, config: NotifyConfig) -> bool:
    """Push message via email."""
    try:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = "期货信号提醒"
        msg["From"] = config.email_sender
        msg["To"] = ", ".join(config.email_recipients)

        with smtplib.SMTP(config.email_smtp, 587) as server:
            server.starttls()
            server.login(config.email_sender, config.email_password)
            server.sendmail(config.email_sender, config.email_recipients, msg.as_string())
        return True
    except Exception as e:
        print(f"Email push failed: {e}")
        return False


def record_signal(symbol: str, direction: int, timestamp: datetime) -> None:
    """Record signal time for cooldown tracking."""
    key = f"{symbol}_{direction}"
    _last_signals[key] = timestamp


def should_push_signal(
    symbol: str,
    direction: int,
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if signal should be pushed (respecting cooldown)."""
    if current_time is None:
        current_time = datetime.now()

    key = f"{symbol}_{direction}"
    last_time = _last_signals.get(key)

    if last_time is None:
        return True

    elapsed = current_time - last_time
    return elapsed.total_seconds() > SIGNAL_COOLDOWN_MINUTES * 60


def push_signal(signal: SignalEvent, config: NotifyConfig) -> Dict[str, bool]:
    """Push signal to all enabled channels."""
    message = format_signal_message(signal)
    results = {}

    if config.wechat_enabled and config.wechat_webhook:
        results["wechat"] = push_wechat(message, config.wechat_webhook)

    if config.telegram_enabled and config.telegram_bot_token:
        results["telegram"] = push_telegram(message, config.telegram_bot_token, config.telegram_chat_id)

    if config.email_enabled:
        results["email"] = push_email(message, config)

    if any(results.values()):
        record_signal(signal.symbol, signal.direction, signal.timestamp)

    return results