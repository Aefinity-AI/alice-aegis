"""Visible synthetic cursor for the X layer.

A tiny always-on-top override-redirect Tk window shaped like an arrow
(bounding shape via the X SHAPE extension — sommelier runs --enable-xshape),
with an EMPTY input shape so real clicks pass straight through it.
Controlled over a unix socket with line-delimited JSON:

  {"cmd":"ping"} {"cmd":"move","x":..,"y":..,"dur":ms|null}
  {"cmd":"clickfx"} {"cmd":"hide"} {"cmd":"show"} {"cmd":"pos"} {"cmd":"quit"}

Run as: python3 -m fablehand.overlay
"""

import json
import os
import socket
import sys
import tkinter as tk

STATE_DIR = os.path.expanduser("~/.fable-hand")
SOCK_PATH = os.path.join(STATE_DIR, "cursor.sock")

W, H = 20, 26  # window size; arrow tip is at (0, 0)
ARROW = [(0, 0), (0, 19), (5, 15), (8, 24), (12, 22), (9, 14), (14, 13)]
CORAL, INK = "#D97757", "#141413"
TICK_MS = 16


def _scanline_rects(poly, w, h):
    """Rasterize the arrow polygon into horizontal rects for shape_rectangles."""
    from PIL import Image, ImageDraw
    img = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=1, outline=1)
    px = img.load()
    rects = []
    for y in range(h):
        x = 0
        while x < w:
            if px[x, y]:
                x0 = x
                while x < w and px[x, y]:
                    x += 1
                rects.append((x0, y, x - x0, 1))
            else:
                x += 1
    return rects


class CursorOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.title("fable-cursor")
        self.canvas = tk.Canvas(self.root, width=W, height=H,
                                highlightthickness=0, bg="white")
        self.canvas.pack()
        self.arrow_id = self.canvas.create_polygon(
            *[c for p in ARROW for c in p],
            fill=CORAL, outline=INK, width=1.5)
        self.x, self.y = 200, 200
        self.hidden = False
        self.anim = None          # (steps list, reply socket)
        self.root.geometry(f"{W}x{H}+{self.x}+{self.y}")
        # Xwayland drops shapes applied before the window is mapped; apply
        # after a full update and re-apply on every map.
        self.root.update()
        self._apply_shape()
        self.root.bind("<Map>", lambda e: self._apply_shape())
        self._serve()

    # -- X SHAPE: arrow-shaped window, click-through input ------------------

    def _apply_shape(self):
        try:
            from Xlib import display as xdisplay
            from Xlib.ext import shape
            d = xdisplay.Display()
            if not d.has_extension("SHAPE"):
                return
            # Tk wraps toplevels: winfo_id() is the inner window; the mapped
            # toplevel the compositor shows is its parent. Shape that one.
            win = d.create_resource_object("window", self.root.winfo_id())
            tree = win.query_tree()
            if tree.parent and tree.parent.id != tree.root.id:
                win = tree.parent
            rects = _scanline_rects(ARROW, W, H)
            win.shape_rectangles(shape.SO.Set, shape.SK.Bounding, 0, 0, 0, rects)
            win.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
            d.sync()
        except Exception as e:
            # Rectangular but functional; hide-during-click still guarantees
            # the real click lands on the target.
            print(f"[overlay] shape not applied: {e}", file=sys.stderr)

    # -- socket server ------------------------------------------------------

    def _serve(self):
        import fcntl
        os.makedirs(STATE_DIR, exist_ok=True)
        # One daemon only: hold an exclusive lock for our whole lifetime so a
        # concurrent start can't unlink the live socket out from under us.
        self._lock = open(os.path.join(STATE_DIR, "overlay.lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[overlay] another daemon holds the lock; exiting",
                  file=sys.stderr)
            sys.exit(0)
        try:
            os.unlink(SOCK_PATH)
        except FileNotFoundError:
            pass
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(SOCK_PATH)
        self.sock.listen(4)
        self.sock.setblocking(False)
        self.root.after(TICK_MS, self._poll)

    def _poll(self):
        try:
            conn, _ = self.sock.accept()
            conn.settimeout(2.0)
            try:
                line = conn.recv(4096).decode().strip()
                if line:
                    self._handle(json.loads(line), conn)
                else:
                    conn.close()
            except Exception:
                conn.close()
        except BlockingIOError:
            pass
        self._step_anim()
        self.root.after(TICK_MS, self._poll)

    @staticmethod
    def _reply(conn, obj):
        if conn is None:
            return
        try:
            conn.sendall((json.dumps(obj) + "\n").encode())
        except Exception:
            pass
        finally:
            conn.close()

    def _handle(self, msg, conn):
        cmd = msg.get("cmd")
        if cmd == "ping":
            self._reply(conn, {"ok": True, "pid": os.getpid()})
        elif cmd == "pos":
            self._reply(conn, {"x": self.x, "y": self.y})
        elif cmd == "hide":
            self.hidden = True
            self.root.withdraw()
            self._reply(conn, {"ok": True})
        elif cmd == "show":
            self.hidden = False
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
            self._apply_shape()
            self._reply(conn, {"ok": True})
        elif cmd == "move":
            self._start_move(int(msg["x"]), int(msg["y"]),
                             msg.get("dur"), conn)
        elif cmd == "clickfx":
            self._clickfx(conn)
        elif cmd == "quit":
            self._reply(conn, {"ok": True})
            self.root.destroy()
        else:
            self._reply(conn, {"error": f"unknown cmd {cmd!r}"})

    # -- animation ----------------------------------------------------------

    def _start_move(self, tx, ty, dur, conn):
        if self.hidden:
            self.hidden = False
            self.root.deiconify()
            self.root.attributes("-topmost", True)
        if self.anim:  # preempt: answer the old caller where we are
            self._reply(self.anim[1], {"x": self.x, "y": self.y,
                                       "preempted": True})
        dist = ((tx - self.x) ** 2 + (ty - self.y) ** 2) ** 0.5
        if dur is None:
            dur = min(700, 150 + dist * 0.5)
        n = max(1, int(dur / TICK_MS))
        x0, y0 = self.x, self.y
        steps = []
        for i in range(1, n + 1):
            t = i / n
            e = 1 - (1 - t) ** 3  # ease-out cubic
            steps.append((round(x0 + (tx - x0) * e), round(y0 + (ty - y0) * e)))
        self.anim = (steps, conn)

    def _step_anim(self):
        if not self.anim:
            return
        steps, conn = self.anim
        self.x, self.y = steps.pop(0)
        self.root.geometry(f"+{self.x}+{self.y}")
        if not steps:
            self.anim = None
            self._reply(conn, {"x": self.x, "y": self.y})

    def _clickfx(self, conn):
        # Squash the arrow briefly: a visible "press".
        self.canvas.scale(self.arrow_id, 0, 0, 0.75, 0.75)
        def restore():
            self.canvas.scale(self.arrow_id, 0, 0, 1 / 0.75, 1 / 0.75)
            self._reply(conn, {"ok": True})
        self.root.after(140, restore)

    def run(self):
        self.root.mainloop()


def main():
    CursorOverlay().run()


if __name__ == "__main__":
    main()
