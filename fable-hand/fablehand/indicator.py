"""Own-mode on-screen indicator: an unmistakable coral strip.

A full-width, always-on-top, override-redirect Tk strip across the top of
the screen reading "FABLE-HAND OWNS THIS BOX". Like the cursor overlay it
gets an EMPTY input shape (X SHAPE extension) so every click passes straight
through it — it can be seen but never intercepts interaction. If SHAPE is
unavailable it shrinks to a small top-right corner badge instead, where it
cannot occlude normal window content.

It never steals focus (override-redirect windows take none) and polls
~/.fable-hand/own.json twice a second, exiting when own mode is inactive.

Run as: python3 -m fablehand.indicator
"""

import fcntl
import os
import signal
import sys
import time
import tkinter as tk

from . import journal

STRIP_H = 26
BADGE_W, BADGE_H = 300, 26  # no-SHAPE fallback size
CORAL, INK = "#D97757", "#141413"
POLL_MS = 500


class OwnIndicator:
    def __init__(self):
        # One indicator only, same flock discipline as the cursor overlay.
        os.makedirs(journal.STATE_DIR, exist_ok=True)
        self._lock = open(os.path.join(journal.STATE_DIR, "indicator.lock"), "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[indicator] another indicator holds the lock; exiting",
                  file=sys.stderr)
            sys.exit(0)

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.title("fable-own")
        sw = self.root.winfo_screenwidth()
        self.w, self.h = sw, STRIP_H
        self.root.geometry(f"{self.w}x{self.h}+0+0")
        self.root.configure(bg=CORAL)

        st = journal._read_own()
        label = st.get("label")
        since = time.strftime("%H:%M:%S", time.localtime(st.get("started",
                                                                time.time())))
        text = f"● FABLE-HAND OWNS THIS BOX — since {since}"
        if label:
            text += f" — {label}"
        text += f" — journal: {journal.JOURNAL_PATH}"
        self.label = tk.Label(self.root, text=text, bg=CORAL, fg=INK,
                              font=("DejaVu Sans", 10, "bold"))
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        self.root.update()
        if not self._make_clickthrough():
            # No SHAPE: never risk intercepting clicks across the whole top
            # edge; become a small corner badge instead.
            self.w, self.h = BADGE_W, BADGE_H
            self.root.geometry(f"{self.w}x{self.h}+{sw - BADGE_W - 8}+4")
            self.label.config(text="● FABLE-HAND OWNS THIS BOX")
        # Xwayland drops shapes applied before mapping; re-apply on every map.
        self.root.bind("<Map>", lambda e: self._make_clickthrough())
        signal.signal(signal.SIGTERM, lambda *a: self.root.after(0, self._quit))
        self.root.after(POLL_MS, self._poll)

    def _make_clickthrough(self):
        """Empty INPUT shape on the mapped wrapper toplevel; True on success."""
        try:
            from Xlib import display as xdisplay
            from Xlib.ext import shape
            d = xdisplay.Display()
            if not d.has_extension("SHAPE"):
                return False
            win = d.create_resource_object("window", self.root.winfo_id())
            tree = win.query_tree()
            if tree.parent and tree.parent.id != tree.root.id:
                win = tree.parent
            win.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
            d.sync()
            return True
        except Exception as e:
            print(f"[indicator] input shape not applied: {e}", file=sys.stderr)
            return False

    def _poll(self):
        if not journal.is_owned():
            self._quit()
            return
        self.root.attributes("-topmost", True)
        self.root.after(POLL_MS, self._poll)

    def _quit(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def main():
    OwnIndicator().run()


if __name__ == "__main__":
    main()
