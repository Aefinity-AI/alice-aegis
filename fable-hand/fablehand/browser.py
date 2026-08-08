"""Chromium control layer: lifecycle, visible in-page cursor, actions.

The browser runs as this container's own Chromium (displays as a normal
ChromeOS window via sommelier) with CDP on 127.0.0.1:9222. Every page gets
an injected animated cursor overlay; actions move the cursor visibly before
dispatching real input events, like the Claude-in-Chrome extension does.
"""

import base64
import json
import os
import shutil
import subprocess
import time

from . import cdp

PORT = cdp.DEFAULT_PORT
STATE_DIR = os.path.expanduser("~/.fable-hand")
PROFILE_DIR = os.path.join(STATE_DIR, "chrome-profile")
CHROME_LOG = os.path.join(STATE_DIR, "chrome.log")
TAB_STATE = os.path.join(STATE_DIR, "current_tab")


def _set_current_tab(target_id):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(TAB_STATE, "w") as f:
        f.write(target_id or "")


def _get_current_tab():
    try:
        with open(TAB_STATE) as f:
            return f.read().strip() or None
    except OSError:
        return None

CURSOR_JS = r"""
(() => {
  if (window.__fableHand) return "already";
  const S = { x: Math.floor(innerWidth/2), y: Math.floor(innerHeight/2) };
  const ARROW = `<svg width="26" height="26" viewBox="0 0 26 26" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 1 L2 19 L7 15 L10.5 23 L14 21.5 L10.5 13.5 L17 13 Z"
          fill="#D97757" stroke="#141413" stroke-width="1.4" stroke-linejoin="round"/>
  </svg>`;
  function ensureStyle() {
    if (document.getElementById('__fable_kf')) return;
    const st = document.createElement('style');
    st.id = '__fable_kf';
    st.textContent = `@keyframes __fable_ripple {
      from { transform: scale(0.4); opacity: 0.9; }
      to   { transform: scale(2.2); opacity: 0; } }`;
    document.documentElement.appendChild(st);
  }
  function ensure() {
    let el = document.getElementById('__fable_cursor');
    if (!el || !el.isConnected) {
      ensureStyle();
      el = document.createElement('div');
      el.id = '__fable_cursor';
      el.style.cssText = 'position:fixed;left:0;top:0;width:26px;height:26px;' +
        'z-index:2147483647;pointer-events:none;margin:0;padding:0;' +
        'transition:transform 350ms cubic-bezier(.25,.7,.3,1);will-change:transform;' +
        'filter:drop-shadow(0 1px 2px rgba(0,0,0,.45));';
      el.innerHTML = ARROW;
      el.style.transform = `translate(${S.x}px,${S.y}px)`;
      (document.body || document.documentElement).appendChild(el);
    }
    return el;
  }
  window.__fableHand = {
    pos: () => ({ x: S.x, y: S.y }),
    moveTo(x, y, dur) {
      const el = ensure();
      const d = (dur === undefined)
        ? Math.min(700, 150 + Math.hypot(x - S.x, y - S.y) * 0.6)
        : dur;
      el.style.transition = `transform ${Math.max(1,d)}ms cubic-bezier(.25,.7,.3,1)`;
      el.style.transform = `translate(${x}px,${y}px)`;
      S.x = x; S.y = y;
      return new Promise(res => setTimeout(res, d + 40));
    },
    clickFx() {
      ensure();
      const r = document.createElement('div');
      r.style.cssText = `position:fixed;left:${S.x-14}px;top:${S.y-14}px;` +
        'width:28px;height:28px;border:3px solid #D97757;border-radius:50%;' +
        'z-index:2147483646;pointer-events:none;' +
        'animation:__fable_ripple 420ms ease-out forwards;';
      (document.body || document.documentElement).appendChild(r);
      setTimeout(() => r.remove(), 500);
      return new Promise(res => setTimeout(res, 120));
    },
  };
  ensure();
  return "installed";
})()
"""


def _chromium_bin():
    for c in ("chromium", "chromium-browser", "google-chrome"):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("no chromium binary found (apt install chromium)")


def ensure_browser(headless=False):
    """Start Chromium with CDP if not already up. Returns True if we started it."""
    if cdp.browser_alive(PORT):
        return False
    os.makedirs(PROFILE_DIR, exist_ok=True)
    args = [
        _chromium_bin(),
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-session-crashed-bubble",
        "--disable-features=TranslateUI",
        "--renderer-process-limit=4",
    ]
    if headless:
        args.append("--headless=new")
    args.append("about:blank")
    env = dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":0"))
    with open(CHROME_LOG, "ab") as log:
        subprocess.Popen(
            args, stdout=log, stderr=log,
            start_new_session=True, env=env,
        )
    deadline = time.time() + 30
    while time.time() < deadline:
        if cdp.browser_alive(PORT):
            return True
        time.sleep(0.3)
    raise SystemExit(f"chromium did not expose CDP on :{PORT}; see {CHROME_LOG}")


def stop_browser():
    if not cdp.browser_alive(PORT):
        return False
    try:
        ver = cdp._http(PORT, "/json/version")
        with cdp.Session(ver["webSocketDebuggerUrl"]) as s:
            s.call("Browser.close")
        return True
    except Exception:
        subprocess.run(["pkill", "-f", f"remote-debugging-port={PORT}"], check=False)
        return True


def _page_session(tab_index=None):
    """Bind to a page: explicit index if given, else the persisted current
    tab by stable targetId (raw /json/list order is newest-first and races
    across invocations), else pages[0] as last resort."""
    pages = cdp.list_pages(PORT)
    if not pages:
        cdp.new_page(port=PORT)
        time.sleep(0.5)
        pages = cdp.list_pages(PORT)
    if tab_index is not None:
        if not 0 <= tab_index < len(pages):
            raise SystemExit(f"tab index {tab_index} out of range "
                             f"(0..{len(pages) - 1})")
        target = pages[tab_index]
    else:
        tid = _get_current_tab()
        target = next((p for p in pages if p.get("id") == tid), pages[0])
    _set_current_tab(target.get("id"))
    sess = cdp.Session(target["webSocketDebuggerUrl"])
    return sess, target


def select_tab(tab_index):
    """Pin the current tab by index, persisted as its stable targetId."""
    pages = cdp.list_pages(PORT)
    if not 0 <= tab_index < len(pages):
        raise SystemExit(f"tab index {tab_index} out of range "
                         f"(0..{len(pages) - 1})")
    _set_current_tab(pages[tab_index].get("id"))
    p = pages[tab_index]
    return {"selected": tab_index, "id": p.get("id"),
            "title": p.get("title"), "url": p.get("url")}


def _install_cursor(sess):
    sess.call("Page.enable")
    sess.call("Page.addScriptToEvaluateOnNewDocument", source=CURSOR_JS)
    sess.eval(CURSOR_JS)


def goto(url, tab_index=None, wait=True, timeout=25):
    ensure_browser()
    if "://" not in url:
        url = "https://" + url
    sess, target = _page_session(tab_index)
    with sess:
        _install_cursor(sess)
        nav = sess.call("Page.navigate", url=url)
        if nav.get("errorText"):
            raise SystemExit(f"navigation failed: {nav['errorText']} ({url})")
        loaded = _wait_load(sess, timeout) if wait else None
        _install_cursor(sess)
        title = sess.eval("document.title") or ""
        cur = sess.eval("location.href")
    out = {"url": cur, "title": title, "targetId": target.get("id")}
    if wait and not loaded:
        out["loaded"] = False
        out["warning"] = f"page did not reach readyState within {timeout}s"
    return out


def _wait_load(sess, timeout=25):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if sess.eval("document.readyState") in ("interactive", "complete"):
                return True
        except cdp.CDPError:
            pass
        time.sleep(0.4)
    return False


def _resolve_target(sess, selector=None, x=None, y=None):
    """Return viewport (x, y) for a CSS selector or passthrough coords."""
    if selector is None:
        return float(x), float(y)
    js = f"""
    (() => {{
      const el = document.querySelector({json.dumps(selector)});
      if (!el) return null;
      el.scrollIntoView({{block:'center', inline:'center', behavior:'instant'}});
      const r = el.getBoundingClientRect();
      return {{x: r.left + r.width/2, y: r.top + r.height/2}};
    }})()
    """
    pt = sess.eval(js)
    if not pt:
        raise SystemExit(f"selector not found: {selector}")
    return float(pt["x"]), float(pt["y"])


def click(selector=None, x=None, y=None, button="left", clicks=1, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        _install_cursor(sess)
        px, py = _resolve_target(sess, selector, x, y)
        sess.eval(f"__fableHand.moveTo({px:.0f}, {py:.0f})", await_promise=True)
        sess.eval("__fableHand.clickFx()", await_promise=True)
        # A double click is press(1)/release(1) then press(2)/release(2);
        # repeating clickCount=2 pairs fires dblclick twice with no first click.
        for count in range(1, clicks + 1):
            for t in ("mousePressed", "mouseReleased"):
                sess.call("Input.dispatchMouseEvent", type=t, x=px, y=py,
                          button=button, clickCount=count)
        time.sleep(0.15)
        return {"clicked": [round(px), round(py)], "selector": selector,
                "url": sess.eval("location.href")}


def type_text(text, selector=None, submit=False, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        _install_cursor(sess)
        if selector:
            px, py = _resolve_target(sess, selector)
            sess.eval(f"__fableHand.moveTo({px:.0f}, {py:.0f})", await_promise=True)
            sess.eval("__fableHand.clickFx()", await_promise=True)
            for t in ("mousePressed", "mouseReleased"):
                sess.call("Input.dispatchMouseEvent", type=t, x=px, y=py,
                          button="left", clickCount=1)
            time.sleep(0.1)
        sess.call("Input.insertText", text=text)
        if submit:
            _press_enter(sess)
        time.sleep(0.1)
        return {"typed": len(text), "selector": selector, "submitted": submit}


def _press_enter(sess):
    # Form submission requires a keyDown that carries text="\r";
    # rawKeyDown is not enough for browser default actions.
    sess.call("Input.dispatchKeyEvent", type="keyDown", key="Enter",
              code="Enter", text="\r", unmodifiedText="\r",
              windowsVirtualKeyCode=13, nativeVirtualKeyCode=13)
    sess.call("Input.dispatchKeyEvent", type="keyUp", key="Enter",
              code="Enter", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13)


def press_key(key_combo, tab_index=None):
    """Dispatch a key chord like 'Enter', 'Tab', 'ctrl+a'."""
    ensure_browser()
    KEYS = {
        "enter": ("Enter", "Enter", 13), "tab": ("Tab", "Tab", 9),
        "escape": ("Escape", "Escape", 27), "backspace": ("Backspace", "Backspace", 8),
        "delete": ("Delete", "Delete", 46), "arrowdown": ("ArrowDown", "ArrowDown", 40),
        "arrowup": ("ArrowUp", "ArrowUp", 38), "arrowleft": ("ArrowLeft", "ArrowLeft", 37),
        "arrowright": ("ArrowRight", "ArrowRight", 39), "pagedown": ("PageDown", "PageDown", 34),
        "pageup": ("PageUp", "PageUp", 33), "home": ("Home", "Home", 36),
        "end": ("End", "End", 35),
    }
    parts = key_combo.lower().split("+")
    mods = 0
    for m in parts[:-1]:
        mods |= {"alt": 1, "ctrl": 2, "control": 2, "meta": 4, "shift": 8}.get(m, 0)
    base = parts[-1]
    if base in KEYS:
        key, code, vk = KEYS[base]
    elif len(base) == 1:
        key, code, vk = base, f"Key{base.upper()}", ord(base.upper())
    else:
        raise SystemExit(f"unknown key: {base}")
    sess, _ = _page_session(tab_index)
    with sess:
        if key == "Enter" and mods == 0:
            _press_enter(sess)
        else:
            for t in ("rawKeyDown", "keyUp"):
                sess.call("Input.dispatchKeyEvent", type=t, key=key, code=code,
                          modifiers=mods, windowsVirtualKeyCode=vk,
                          nativeVirtualKeyCode=vk)
    return {"pressed": key_combo}


def scroll(dy=600, dx=0, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        _install_cursor(sess)
        pos = sess.eval("__fableHand.pos()")
        sess.call("Input.dispatchMouseEvent", type="mouseWheel",
                  x=pos["x"], y=pos["y"], deltaX=dx, deltaY=dy)
        time.sleep(0.2)
        return {"scrolled": [dx, dy]}


def screenshot(out_path, full_page=False, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        _install_cursor(sess)
        kwargs = {"format": "png"}
        if full_page:
            kwargs["captureBeyondViewport"] = True
        res = sess.call("Page.captureScreenshot", **kwargs)
    data = base64.b64decode(res["data"])
    with open(out_path, "wb") as f:
        f.write(data)
    return {"path": out_path, "bytes": len(data)}


def read_page(selector=None, max_chars=20000, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        if selector:
            src = f"document.querySelector({json.dumps(selector)})?.innerText"
        else:
            src = "(document.body ? document.body.innerText : '')"
        # Truncate inside the page so a huge document can't blow the CDP frame.
        js = (f"(t => t == null ? null : "
              f"{{text: t.slice(0, {int(max_chars)}), total: t.length}})({src})")
        res = sess.eval(js)
        info = {"url": sess.eval("location.href"),
                "title": sess.eval("document.title")}
    if res is None:
        raise SystemExit(f"selector not found: {selector}")
    info["text"] = res["text"]
    info["truncated"] = res["total"] > max_chars
    return info


def eval_js(expression, tab_index=None):
    ensure_browser()
    sess, _ = _page_session(tab_index)
    with sess:
        return sess.eval(expression, await_promise=True)


def tabs():
    if not cdp.browser_alive(PORT):
        return []
    cur = _get_current_tab()
    return [{"i": i, "title": p.get("title"), "url": p.get("url"),
             "id": p.get("id"), "current": p.get("id") == cur}
            for i, p in enumerate(cdp.list_pages(PORT))]
