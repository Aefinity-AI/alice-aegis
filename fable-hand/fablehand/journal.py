"""Action journal + own-mode ("own the box") state.

The journal is ALWAYS on: every fable-hand action appends one JSON line to
~/.fable-hand/journal.jsonl as {ts, verb, args, result|error, owned}. Own
mode does not enable journaling — it flags entries owned:true and puts an
unmistakable indicator on screen (fablehand/indicator.py) so the box's
operator can see, and later replay, exactly what the agent did.

Writes are crash-safe: one line per action, append + flush + fsync.
"""

import json
import os
import subprocess
import sys
import time

STATE_DIR = os.path.expanduser("~/.fable-hand")
JOURNAL_PATH = os.path.join(STATE_DIR, "journal.jsonl")
OWN_STATE = os.path.join(STATE_DIR, "own.json")
INDICATOR_LOG = os.path.join(STATE_DIR, "indicator.log")

MAX_STR = 4096  # journal stores actions, not payload dumps: cap long strings


def _iso(ts=None):
    t = time.time() if ts is None else ts
    lt = time.localtime(t)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f".{int(t * 1000) % 1000:03d}"


def _compact(obj):
    """JSON-safe copy with long strings truncated (audit trail, not payload)."""
    if isinstance(obj, str):
        if len(obj) > MAX_STR:
            return obj[:MAX_STR] + f"...[truncated {len(obj) - MAX_STR} chars]"
        return obj
    if isinstance(obj, dict):
        return {str(k): _compact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_compact(v) for v in obj]
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return _compact(str(obj))


def append(verb, args, result=None, error=None, extra=None):
    """Crash-safe single-line append. Never raises into the action path."""
    entry = {"ts": _iso(), "verb": verb, "args": _compact(args),
             "owned": is_owned()}
    if error is not None:
        entry["error"] = _compact(error)
    else:
        entry["result"] = _compact(result)
    if extra:
        entry.update(_compact(extra))
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(JOURNAL_PATH, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        print(f"[journal warning] {e}", file=sys.stderr)


# -- own mode ----------------------------------------------------------------

def _read_own():
    try:
        with open(OWN_STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_own(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = OWN_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OWN_STATE)


def is_owned():
    return _read_own().get("active") is True


def _indicator_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _count_actions(since_ts):
    """Owned, non-'own' journal entries since `since_ts` (epoch seconds)."""
    n = 0
    since_iso = _iso(since_ts)
    try:
        with open(JOURNAL_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if (e.get("owned") and e.get("verb") != "own"
                        and e.get("ts", "") >= since_iso):
                    n += 1
    except OSError:
        pass
    return n


def own_start(label=None):
    st = _read_own()
    if st.get("active") and _indicator_alive(st.get("indicator_pid")):
        raise SystemExit(
            f"own mode already active since {_iso(st.get('started', 0))}"
            + (f" (label {st['label']!r})" if st.get("label") else ""))
    state = {"active": True, "started": time.time(), "label": label}
    _write_own(state)
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(INDICATOR_LOG, "ab") as log:
        p = subprocess.Popen(
            [sys.executable, "-m", "fablehand.indicator"],
            stdout=log, stderr=log, start_new_session=True,
            env=dict(os.environ,
                     DISPLAY=os.environ.get("DISPLAY", ":0"),
                     PYTHONPATH=os.path.dirname(os.path.dirname(
                         os.path.abspath(__file__)))),
        )
    state["indicator_pid"] = p.pid
    _write_own(state)
    # The indicator is the point: wait until its X window is actually mapped.
    deadline = time.time() + 8
    visible = False
    while time.time() < deadline:
        res = subprocess.run(
            ["xdotool", "search", "--name", "^fable-own$"],
            capture_output=True, text=True,
            env=dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0")))
        if res.stdout.strip():
            visible = True
            break
        time.sleep(0.2)
    out = {"own": "started", "label": label, "started": _iso(state["started"]),
           "indicator_pid": p.pid, "indicator_visible": visible,
           "journal": JOURNAL_PATH}
    if not visible:
        out["warning"] = f"indicator window not seen within 8s; see {INDICATOR_LOG}"
    return out


def own_stop():
    st = _read_own()
    if not st.get("active"):
        raise SystemExit("own mode is not active")
    started = st.get("started", time.time())
    n_actions = _count_actions(started)
    _write_own({"active": False, "stopped": time.time(),
                "label": st.get("label"), "started": started})
    # Indicator polls own.json and exits on its own; nudge it for promptness.
    pid = st.get("indicator_pid")
    if _indicator_alive(pid):
        try:
            os.kill(int(pid), 15)
        except OSError:
            pass
    return {"own": "stopped", "label": st.get("label"),
            "started": _iso(started), "duration_s": round(time.time() - started, 1),
            "n_actions": n_actions, "journal": JOURNAL_PATH}


def own_status():
    st = _read_own()
    if not st.get("active"):
        return {"active": False, "journal": JOURNAL_PATH}
    started = st.get("started", 0)
    return {"active": True, "label": st.get("label"),
            "started": _iso(started),
            "duration_s": round(time.time() - started, 1),
            "n_actions": _count_actions(started),
            "indicator_alive": _indicator_alive(st.get("indicator_pid")),
            "journal": JOURNAL_PATH}
