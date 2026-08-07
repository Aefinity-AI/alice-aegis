"""Minimal Chrome DevTools Protocol client.

Uses the sync websocket client from python3-websockets (>=12). One Session
per page target; call() is strictly request/response, events are collected
into .events for callers that care.
"""

import json
import time
import urllib.request
import urllib.parse

try:
    from websockets.sync.client import connect as _ws_connect
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "fable-hand needs python3-websockets >= 12 (sync client). "
        f"Import failed: {e}"
    )

DEFAULT_PORT = 9222
MAX_MSG = 128 * 1024 * 1024  # screenshots come back base64 in one frame


class CDPError(Exception):
    pass


def _http(port, path, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        body = r.read()
    return json.loads(body) if body else {}


def browser_alive(port=DEFAULT_PORT):
    try:
        return "webSocketDebuggerUrl" in _http(port, "/json/version")
    except Exception:
        return False


def list_pages(port=DEFAULT_PORT):
    return [t for t in _http(port, "/json/list") if t.get("type") == "page"]


def new_page(url="about:blank", port=DEFAULT_PORT):
    q = urllib.parse.quote(url, safe=":/?&=%#")
    return _http(port, f"/json/new?{q}", method="PUT")


def close_page(target_id, port=DEFAULT_PORT):
    return _http(port, f"/json/close/{target_id}")


def activate_page(target_id, port=DEFAULT_PORT):
    return _http(port, f"/json/activate/{target_id}")


class Session:
    """Sync CDP session bound to one page target."""

    def __init__(self, ws_url, timeout=30):
        self.ws = _ws_connect(ws_url, max_size=MAX_MSG, open_timeout=10)
        self.timeout = timeout
        self.events = []
        self._id = 0

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def call(self, method, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        # Overall deadline: a steady event stream must not extend the wait.
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CDPError(f"{method}: no response within {self.timeout}s")
            msg = json.loads(self.ws.recv(timeout=remaining))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise CDPError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    # -- conveniences -------------------------------------------------------

    def eval(self, expression, await_promise=False, return_by_value=True):
        res = self.call(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=await_promise,
            returnByValue=return_by_value,
            userGesture=True,
        )
        if res.get("exceptionDetails"):
            raise CDPError(f"JS exception: {res['exceptionDetails'].get('text')} "
                           f"{res['exceptionDetails'].get('exception', {}).get('description', '')}")
        return res.get("result", {}).get("value")


def open_session(target=None, port=DEFAULT_PORT, timeout=30):
    """Open a Session to `target` (dict from list_pages) or the first page."""
    if target is None:
        pages = list_pages(port)
        if not pages:
            target = new_page(port=port)
        else:
            target = pages[0]
    return Session(target["webSocketDebuggerUrl"], timeout=timeout), target
