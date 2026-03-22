"""Tests for notify module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from autoresearch_futures.notify import (
    format_signal_message,
    push_signal,
    should_push_signal,
    NotifyConfig,
    SignalEvent,
)


class TestNotify:
    @pytest.fixture
    def config(self):
        return NotifyConfig(
            wechat_enabled=True,
            wechat_webhook="https://example.com/webhook",
            telegram_enabled=False,
            email_enabled=False,
        )

    @pytest.fixture
    def signal_event(self):
        return SignalEvent(
            symbol="rb2405",
            direction=1,
            timestamp=datetime(2024, 3, 23, 14, 30),
            price=3650.0,
            confidence=0.75,
            sources=["SMC: Order Block 支撑", "动能: RSI 超卖反弹"],
            suggested_stop=3620.0,
            suggested_target=3720.0,
        )

    def test_format_signal_message(self, signal_event):
        """format_signal_message should format correctly."""
        msg = format_signal_message(signal_event)
        assert "rb2405" in msg
        assert "做多" in msg
        assert "3650" in msg
        assert "75%" in msg

    @patch("requests.post")
    def test_push_signal_wechat(self, mock_post, signal_event, config):
        """push_signal should call wechat webhook."""
        mock_post.return_value = MagicMock(status_code=200)
        push_signal(signal_event, config)
        assert mock_post.called

    def test_should_push_signal_cooldown(self):
        """should_push_signal should respect cooldown."""
        from autoresearch_futures.notify import record_signal, SIGNAL_COOLDOWN_MINUTES
        # Clear any existing record
        record_signal("rb_test", 1, datetime.now() - timedelta(hours=1))

        # Should allow after cooldown
        future_time = datetime.now() + timedelta(minutes=SIGNAL_COOLDOWN_MINUTES + 1)
        assert should_push_signal("rb_test", 1, current_time=future_time) is True