"""
Django WebSocket integration for real-time training metrics.
Sends training metrics to Django dashboard via WebSocket or HTTP API.
"""

import json
import asyncio
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Container for training metrics to send to Django."""
    epoch: int
    batch: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    perplexity: Optional[float] = None
    learning_rate: Optional[float] = None
    gpu_memory: Optional[float] = None
    gpu_utilization: Optional[float] = None
    claude_suggestion: Optional[str] = None


class DjangoMetricsReporter:
    """Reports training metrics to Django dashboard."""

    def __init__(self,
                 websocket_url: str = "ws://localhost:8000/ws/training/global/",
                 http_url: str = "http://localhost:8000/training/api/metrics/",
                 session_id: Optional[int] = None):
        """
        Initialize Django metrics reporter.

        Args:
            websocket_url: WebSocket endpoint for real-time updates
            http_url: HTTP API endpoint as fallback
            session_id: Django training session ID
        """
        self.websocket_url = websocket_url
        self.http_url = http_url
        self.session_id = session_id
        self.ws = None
        self.ws_thread = None
        self.is_connected = False

        # Try to connect WebSocket if available
        if WEBSOCKET_AVAILABLE:
            self._connect_websocket()

    def _connect_websocket(self):
        """Connect to Django WebSocket."""
        try:
            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            # Run WebSocket in a separate thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                daemon=True
            )
            self.ws_thread.start()

            # Wait briefly for connection
            import time
            time.sleep(0.5)

        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            self.is_connected = False

    def _on_open(self, ws):
        """WebSocket connection opened."""
        self.is_connected = True
        print("Connected to Django dashboard via WebSocket")

    def _on_message(self, ws, message):
        """Handle messages from Django."""
        try:
            data = json.loads(message)
            # Handle control messages if needed
            if data.get('type') == 'stop_training':
                # Signal to stop training
                pass
        except Exception as e:
            print(f"Error handling message: {e}")

    def _on_error(self, ws, error):
        """WebSocket error occurred."""
        print(f"WebSocket error: {error}")
        self.is_connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed."""
        self.is_connected = False
        print("Disconnected from Django dashboard")

    def send_metrics(self, metrics: TrainingMetrics):
        """
        Send training metrics to Django.

        Args:
            metrics: TrainingMetrics object with current values
        """
        # Convert metrics to dict
        metrics_dict = {
            'type': 'metrics_update',
            'metrics': {
                'epoch': metrics.epoch,
                'batch': metrics.batch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'perplexity': metrics.perplexity,
                'learning_rate': metrics.learning_rate,
                'gpu_memory': metrics.gpu_memory,
                'gpu_utilization': metrics.gpu_utilization,
                'claude_suggestion': metrics.claude_suggestion
            }
        }

        # Remove None values
        metrics_dict['metrics'] = {
            k: v for k, v in metrics_dict['metrics'].items()
            if v is not None
        }

        # Add session ID if available
        if self.session_id:
            metrics_dict['session_id'] = self.session_id

        # Try WebSocket first
        if self.is_connected and self.ws:
            try:
                self.ws.send(json.dumps(metrics_dict))
                return
            except Exception as e:
                print(f"WebSocket send failed: {e}")
                self.is_connected = False

        # Fallback to HTTP API
        self._send_via_http(metrics_dict)

    def _send_via_http(self, metrics_dict: Dict[str, Any]):
        """Send metrics via HTTP API as fallback."""
        try:
            response = requests.post(
                self.http_url,
                json=metrics_dict,
                timeout=1  # Short timeout to not block training
            )
            if response.status_code != 200:
                print(f"HTTP metrics send failed: {response.status_code}")
        except requests.exceptions.RequestException:
            # Silently fail - don't interrupt training
            pass

    def send_status(self, status: str, message: Optional[str] = None):
        """
        Send training status update.

        Args:
            status: Status string ('running', 'completed', 'failed', 'paused')
            message: Optional status message
        """
        status_dict = {
            'type': 'status_update',
            'status': status,
            'message': message
        }

        if self.session_id:
            status_dict['session_id'] = self.session_id

        if self.is_connected and self.ws:
            try:
                self.ws.send(json.dumps(status_dict))
            except:
                pass

    def send_claude_suggestion(self, suggestion: str, cost: float = 0.0):
        """
        Send Claude AI suggestion.

        Args:
            suggestion: Claude's suggestion text
            cost: API call cost
        """
        suggestion_dict = {
            'type': 'claude_suggestion',
            'suggestion': suggestion,
            'cost': cost
        }

        if self.session_id:
            suggestion_dict['session_id'] = self.session_id

        if self.is_connected and self.ws:
            try:
                self.ws.send(json.dumps(suggestion_dict))
            except:
                pass

    def close(self):
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.is_connected = False


# Global reporter instance
_reporter: Optional[DjangoMetricsReporter] = None


def initialize_django_reporter(session_id: Optional[int] = None,
                              websocket_url: Optional[str] = None):
    """
    Initialize global Django reporter.

    Args:
        session_id: Django training session ID
        websocket_url: Custom WebSocket URL
    """
    global _reporter

    if websocket_url is None:
        websocket_url = "ws://localhost:8000/ws/training/global/"

    _reporter = DjangoMetricsReporter(
        websocket_url=websocket_url,
        session_id=session_id
    )
    return _reporter


def get_django_reporter() -> Optional[DjangoMetricsReporter]:
    """Get the global Django reporter instance."""
    return _reporter


def send_training_metrics(epoch: int,
                         train_loss: float,
                         val_loss: Optional[float] = None,
                         perplexity: Optional[float] = None,
                         **kwargs):
    """
    Convenience function to send training metrics.

    Args:
        epoch: Current epoch number
        train_loss: Training loss value
        val_loss: Validation loss value
        perplexity: Model perplexity
        **kwargs: Additional metric fields
    """
    if _reporter:
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            **kwargs
        )
        _reporter.send_metrics(metrics)