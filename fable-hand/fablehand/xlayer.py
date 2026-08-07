"""X action layer: XTEST input + window capture for container X clients.

Wraps xdotool (XTEST confirmed present on :0, sommelier Xwayland). The
visible synthetic cursor is a separate overlay daemon (overlay.py); actions
here animate it to the target, hide it for the instant of the click so the
click lands on the real target, then show it again.
"""

import json
import os
import socket
import subprocess
import sys
import time

DISPLAY = os.environ.get("DISPLAY", ":0")
STATE_DIR = os.path.expanduser("~/.fable-hand")
CURSOR_SOCK = os.path.join(STATE_DIR, "cursor.sock")
OVERLAY_LOG = os.path.join(STATE_DIR, "overlay.log")


def _xdo(*args, check=True):
    res = subprocess.run(
        ["xdotool", *args],
        capture_output=True, text=True,
        env=dict(os.environ, DISPLAY=DISPLAY),
    )
    if check and res.returncode != 0:
        raise SystemExit(f"xdotool {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout.strip()


# -- overlay client ---------------------------------------------------------

def _overlay_send(msg, timeout=5.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(CURSOR_SOCK)
        s.sendall((json.dumps(msg) + "\n").encode())
        return json.loads(s.recv(4096).decode() or "{}")
    finally:
        s.close()


def overlay_alive():
    try:
        return _overlay_send({"cmd": "ping"}).get("ok") is True
    except Exception:
        return False


def ensure_overlay():
    if overlay_alive():
        return False
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(OVERLAY_LOG, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "fablehand.overlay"],
            stdout=log, stderr=log, start_new_session=True,
            env=dict(os.environ, DISPLAY=DISPLAY,
                     PYTHONPATH=os.path.dirname(os.path.dirname(__file__))),
        )
    deadline = time.time() + 8
    while time.time() < deadline:
        if overlay_alive():
            return True
        time.sleep(0.2)
    raise SystemExit(f"cursor overlay daemon did not start; see {OVERLAY_LOG}")


def overlay_move(x, y, dur=None, visible_action=True):
    """Animate the visible cursor to (x, y). Best-effort: never blocks actions."""
    if not visible_action:
        return
    try:
        ensure_overlay()
        _overlay_send({"cmd": "move", "x": int(x), "y": int(y),
                       "dur": dur}, timeout=8.0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[overlay warning] {e}", file=sys.stderr)


# -- windows ----------------------------------------------------------------

def windows():
    out = _xdo("search", "--onlyvisible", "--name", ".", check=False)
    result = []
    for wid in out.split():
        name = _xdo("getwindowname", wid, check=False)
        geo = _xdo("getwindowgeometry", "--shell", wid, check=False)
        g = {}
        for line in geo.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                g[k.lower()] = v
        result.append({
            "id": wid, "name": name,
            "x": int(g.get("x", 0)), "y": int(g.get("y", 0)),
            "w": int(g.get("width", 0)), "h": int(g.get("height", 0)),
        })
    return result


def find_window(needle):
    """Resolve a window: numeric id, exact name, or UNIQUE name substring.
    Returns the window dict (id + geometry) so callers resolve exactly once."""
    if needle.isdigit():
        # Numeric ids resolve against ALL windows (including the cursor
        # overlay itself — capturing it by id is legitimate).
        for w in windows():
            if w["id"] == needle:
                return w
        g = _xdo("getwindowgeometry", "--shell", needle, check=False)
        d = dict(line.split("=", 1) for line in g.splitlines() if "=" in line)
        if not d:
            raise SystemExit(f"no window with id {needle}")
        return {"id": needle, "name": None,
                "x": int(d.get("X", 0)), "y": int(d.get("Y", 0)),
                "w": int(d.get("WIDTH", 0)), "h": int(d.get("HEIGHT", 0))}
    wins = [w for w in windows()
            if w["name"] not in ("fable-cursor", "fable-own")]
    exact = [w for w in wins if (w["name"] or "") == needle]
    if len(exact) == 1:
        return exact[0]
    matches = [w for w in wins
               if needle.lower() in (w["name"] or "").lower()]
    if not matches:
        raise SystemExit(f"no visible window matching {needle!r}")
    if len(matches) > 1:
        names = ", ".join(f"{w['id']}:{w['name']!r}" for w in matches)
        raise SystemExit(
            f"ambiguous window {needle!r} matches: {names} — use the id")
    return matches[0]


def activate(window_id):
    _xdo("windowactivate", "--sync", window_id, check=False)
    time.sleep(0.15)


# -- actions ----------------------------------------------------------------

def _resolve(x, y, window=None):
    """Resolve (window-relative or global) coords ONCE; returns (gx, gy, win)."""
    if window is None:
        return int(x), int(y), None
    w = find_window(window)
    return w["x"] + int(x), w["y"] + int(y), w


def move(x, y, window=None):
    gx, gy, _ = _resolve(x, y, window)
    overlay_move(gx, gy)
    _xdo("mousemove", "--sync", "--", str(gx), str(gy))
    return {"moved": [gx, gy]}


def click(x, y, window=None, button=1, double=False):
    gx, gy, w = _resolve(x, y, window)
    if w:
        activate(w["id"])
    overlay_move(gx, gy)
    try:
        _overlay_send({"cmd": "clickfx"})
        _overlay_send({"cmd": "hide"})
    except Exception:
        pass
    try:
        _xdo("mousemove", "--sync", "--", str(gx), str(gy))
        n = "--repeat=2" if double else "--repeat=1"
        _xdo("click", n, str(button))
    finally:
        # Whatever happened, never leave the visible cursor hidden.
        try:
            _overlay_send({"cmd": "show"})
        except Exception:
            pass
    return {"clicked": [gx, gy], "button": button, "double": double}


def type_text(text, window=None, delay_ms=12):
    if window:
        activate(find_window(window)["id"])
    _xdo("type", "--clearmodifiers", "--delay", str(delay_ms), "--", text)
    return {"typed": len(text), "window": window}


def key(combo, window=None):
    if window:
        activate(find_window(window)["id"])
    _xdo("key", "--clearmodifiers", "--", combo)
    return {"pressed": combo}


# -- capture / launch -------------------------------------------------------

def shot(out_path, window=None):
    """Capture one window (rootless X: the root has no composited content)."""
    env = dict(os.environ, DISPLAY=DISPLAY)
    if window:
        wid = find_window(window)["id"]
        res = subprocess.run(["import", "-window", wid, out_path],
                             capture_output=True, text=True, env=env)
    else:
        res = subprocess.run(["scrot", "--overwrite", out_path],
                             capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise SystemExit(f"capture failed: {res.stderr.strip()}")
    return {"path": out_path, "bytes": os.path.getsize(out_path),
            "window": window}


def launch(cmd_list):
    """Detached spawn (setsid via start_new_session); output to launch.log,
    which lives next to the journal in ~/.fable-hand/."""
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, "launch.log"), "ab") as log:
        p = subprocess.Popen(cmd_list, stdout=log, stderr=log,
                             start_new_session=True,
                             env=dict(os.environ, DISPLAY=DISPLAY))
    return {"launched": cmd_list, "pid": p.pid,
            "log": os.path.join(STATE_DIR, "launch.log")}


# -- window management (v2) -------------------------------------------------

def _geometry(window_id):
    g = _xdo("getwindowgeometry", "--shell", window_id, check=False)
    d = dict(line.split("=", 1) for line in g.splitlines() if "=" in line)
    return {"x": int(d.get("X", 0)), "y": int(d.get("Y", 0)),
            "w": int(d.get("WIDTH", 0)), "h": int(d.get("HEIGHT", 0))}


def win_list():
    result = []
    for w in windows():
        desk = _xdo("get_desktop_for_window", w["id"], check=False)
        w["desktop"] = int(desk) if desk.lstrip("-").isdigit() else None
        result.append(w)
    return result


def win_focus(sel):
    w = find_window(sel)
    _xdo("windowactivate", "--sync", w["id"], check=False)
    _xdo("windowraise", w["id"], check=False)
    return {"focused": w["id"], "name": w["name"]}


def win_close(sel):
    w = find_window(sel)
    _xdo("windowclose", w["id"])
    return {"closed": w["id"], "name": w["name"]}


def win_move(sel, x, y):
    w = find_window(sel)
    # No --sync: a WM/compositor that vetoes the move would hang it forever.
    # Requery instead and report where the window ACTUALLY ended up.
    _xdo("windowmove", w["id"], str(int(x)), str(int(y)))
    time.sleep(0.25)
    geo = _geometry(w["id"])
    out = {"id": w["id"], "name": w["name"], "requested": [int(x), int(y)],
           "geometry": geo}
    if [geo["x"], geo["y"]] != [int(x), int(y)]:
        # Measured on this box 2026-08-01: sommelier forwards size changes
        # but toplevel PLACEMENT belongs to the host (ChromeOS) WM.
        out["note"] = ("host WM controls toplevel placement under sommelier; "
                       "request sent, actual geometry reported")
    return out


def win_resize(sel, width, height):
    w = find_window(sel)
    _xdo("windowsize", w["id"], str(int(width)), str(int(height)))
    time.sleep(0.25)
    return {"id": w["id"], "name": w["name"],
            "requested": [int(width), int(height)],
            "geometry": _geometry(w["id"])}


def win_minimize(sel):
    w = find_window(sel)
    _xdo("windowminimize", w["id"])
    return {"minimized": w["id"], "name": w["name"]}


def win_maximize(sel):
    """No EWMH maximize in this xdotool build: move to origin + resize to
    the display. Reported honestly as method 'move+resize'."""
    w = find_window(sel)
    geo = _xdo("getdisplaygeometry").split()
    sw, sh = int(geo[0]), int(geo[1])
    _xdo("windowmove", w["id"], "0", "0")
    _xdo("windowsize", w["id"], str(sw), str(sh))
    time.sleep(0.25)
    return {"maximized": w["id"], "name": w["name"],
            "method": "move+resize", "geometry": _geometry(w["id"])}


# -- .desktop applications (v2) ---------------------------------------------

def apps():
    """Minimal .desktop scan: Name/Exec/Comment from the standard dirs."""
    dirs = ["/usr/share/applications",
            os.path.expanduser("~/.local/share/applications")]
    result = []
    for d in dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            if not fn.endswith(".desktop"):
                continue
            path = os.path.join(d, fn)
            entry, in_section = {}, False
            try:
                with open(path, errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("["):
                            in_section = line == "[Desktop Entry]"
                        elif in_section and "=" in line:
                            k, v = line.split("=", 1)
                            if k in ("Name", "Exec", "Comment", "NoDisplay",
                                     "Terminal") and k not in entry:
                                entry[k] = v
            except OSError:
                continue
            if entry.get("NoDisplay", "").lower() == "true":
                continue
            if not entry.get("Name") or not entry.get("Exec"):
                continue
            item = {"name": entry["Name"], "exec": entry["Exec"], "file": path}
            if entry.get("Comment"):
                item["comment"] = entry["Comment"]
            if entry.get("Terminal", "").lower() == "true":
                item["terminal"] = True
            result.append(item)
    return result


# -- clipboard (v2) ----------------------------------------------------------

def clip_get():
    res = subprocess.run(
        ["xclip", "-selection", "clipboard", "-out"],
        capture_output=True, text=True,
        env=dict(os.environ, DISPLAY=DISPLAY))
    if res.returncode != 0:
        raise SystemExit(f"clipboard read failed: "
                         f"{res.stderr.strip() or 'clipboard empty?'}")
    return {"text": res.stdout, "chars": len(res.stdout)}


def clip_set(text):
    # xclip forks to keep serving the selection; capturing its stdout/stderr
    # would block on the inherited pipe fds, so send them to /dev/null.
    res = subprocess.run(
        ["xclip", "-selection", "clipboard", "-in"],
        input=text.encode(), stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ, DISPLAY=DISPLAY))
    if res.returncode != 0:
        raise SystemExit("clipboard write failed (xclip -in)")
    return {"set": True, "chars": len(text)}
