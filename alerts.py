"""
alerts.py — Live violation alert broadcaster (Server-Sent Events).

Any number of browser tabs can subscribe. Each subscriber gets its own
Queue; publish() drops a copy of the alert into every subscriber's queue.
Dead/disconnected subscribers are cleaned up automatically when their
queue write fails or the SSE generator exits.
"""

import json
import queue
import threading
import time

_subscribers: set = set()
_lock = threading.Lock()


def subscribe() -> "queue.Queue":
    q = queue.Queue()
    with _lock:
        _subscribers.add(q)
    return q


def unsubscribe(q: "queue.Queue") -> None:
    with _lock:
        _subscribers.discard(q)


def publish(alert: dict) -> None:
    """alert should be a small JSON-serializable dict."""
    with _lock:
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(alert)
        except Exception:
            pass


def stream():
    """Generator for the Flask SSE route."""
    q = subscribe()
    try:
        while True:
            try:
                alert = q.get(timeout=15)
                yield f"data: {json.dumps(alert)}\n\n"
            except queue.Empty:
                # heartbeat comment keeps the connection alive through proxies
                yield ": heartbeat\n\n"
    finally:
        unsubscribe(q)