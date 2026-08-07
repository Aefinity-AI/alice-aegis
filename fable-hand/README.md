# fable-hand

Fable 5's self-made computer-use + web-browser utility for this machine: a
local analog of the Claude-in-Chrome extension, driven from the CLI, with a
**visible synthetic cursor** whenever it acts.

Built 2026-08-01 inside the Crostini container "penguin" (Debian 13) on the
i5-10210U Chromebook. v2 ("own the box") added the same day: takeover mode
with an on-screen indicator, an always-on audit journal, window management,
app launch/list, clipboard, and a script runner.

## Honest scope

This tool can act on:

- **Any app running in this Linux container.** Container apps display as
  normal ChromeOS windows (sommelier forwards them), so actions here are
  visible on screen. Input goes through XTEST on `:0`.
- **The full web**, through the tool's own Chromium (installed in the
  container, shown as a normal window), driven over the Chrome DevTools
  Protocol.

It can **not** inject input into the ChromeOS host UI itself (host Chrome
windows, the shelf, other Android/host apps). No guest→host input path
exists from this container: `/dev/uinput` has no consumer here and sommelier
only forwards output, not synthetic host input. "Any action on the PC" is
therefore honestly scoped to: container apps + the web.

## The visible cursor

- **Browser layer:** every page gets an injected coral arrow
  (`window.__fableHand`) that animates to the target with ease-out motion and
  pulses on click, before real CDP input events are dispatched — the same
  pattern the Claude-in-Chrome extension uses.
- **X layer:** a daemon (`fablehand/overlay.py`) shows an always-on-top,
  arrow-**shaped** window (X SHAPE extension, applied to the Tk wrapper
  toplevel) with an **empty input shape**, so it is completely click-through.
  It animates to the target; during the actual click it hides for ~150 ms as
  a second guarantee that the click lands on the real target.

## Usage

```bash
fable-hand status                      # what's alive + scope line
# X layer
fable-hand windows
fable-hand launch xclock -digital
fable-hand shot out.png --window xclock
fable-hand move 1200 800
fable-hand click 300 200 --window xclock
fable-hand type "text" --window NAME
fable-hand key ctrl+l --window NAME
# browser layer
fable-hand browse https://example.com
fable-hand b-click "#submit"           # or: b-click --at 640 360
fable-hand b-type "query" --selector "input[name=q]" --submit
fable-hand b-key Enter
fable-hand b-scroll --dy 800
fable-hand b-read --selector "main"
fable-hand b-shot page.png --full
fable-hand b-eval "document.title"
fable-hand b-tabs                      # marks the current (pinned) tab
fable-hand b-tab 1                     # pin tab 1 as current, by stable id
fable-hand stop-browser
```

All output is JSON on stdout; failures are JSON `{"error": ...}` with exit 1
(failed navigations included — `net::ERR_*` is surfaced, never swallowed).
Tab addressing is by stable targetId persisted in `~/.fable-hand/current_tab`,
so a popup appearing at the front of `/json/list` cannot steal subsequent
commands. Reviewed by a 24-agent adversarial workflow 2026-08-01; all 10
confirmed findings fixed and regression-tested the same day. State lives in `~/.fable-hand/` (Chromium
profile, cursor socket, logs).

## v2 — "own the box"

### Own mode + audit journal

```bash
fable-hand own start --label "task name"   # indicator strip appears
fable-hand own status                       # active?, action count so far
fable-hand own stop                         # session summary
```

While own mode is active, a full-width **coral strip across the top of the
screen** reads `● FABLE-HAND OWNS THIS BOX — since HH:MM:SS — label —
journal: …`. It is an always-on-top override-redirect Tk window
(`fablehand/indicator.py`, same pattern as the cursor overlay) with an
**empty X SHAPE input region**, verified by read-back: it can be seen but
never intercepts a click and never takes focus. If SHAPE were unavailable it
falls back to a small top-right corner badge. The indicator daemon polls
`~/.fable-hand/own.json` and removes itself when own mode ends.

**The journal is always on** — own mode does not enable it, it only flags
entries `owned:true`. Every CLI action appends one line to
`~/.fable-hand/journal.jsonl` (append + flush + fsync, one line per action,
crash-safe):

```json
{"ts":"2026-08-01T19:49:00.138","verb":"move","args":{"x":700,"y":500},
 "owned":true,"result":{"moved":[700,500]}}
```

Fields: `ts` (local ISO, ms), `verb`, `args` (parsed CLI args), exactly one
of `result` | `error`, `owned` (bool). Script-runner steps additionally carry
`via:"run"`, `script`, `step`. Strings longer than 4096 chars are truncated
with a marker — the journal is an audit trail, not a payload store. `own
stop` prints `{n_actions, duration_s, journal}` where `n_actions` counts
owned entries (excluding the `own` verbs themselves); the operator can replay
a session by feeding journal entries back through `fable-hand run`.

### Window management

```bash
fable-hand win list                # id, desktop, geometry, title
fable-hand win focus SEL           # SEL = id | exact title | UNIQUE substring
fable-hand win close SEL           # ambiguity -> error listing matches
fable-hand win move SEL X Y
fable-hand win resize SEL 800x600
fable-hand win minimize SEL
fable-hand win maximize SEL        # move+resize to display (no EWMH here)
```

All backed by xdotool. Honesty note, measured on this box: **sommelier
forwards size changes but the host (ChromeOS) WM owns toplevel placement** —
`win resize` is honored exactly; `win move` sends the request and reports the
window's *actual* resulting geometry, with a `note` when placement was
vetoed. `win maximize` is `move 0,0 + resize to display` because this
xdotool has no EWMH maximize; the result reports `method: "move+resize"`.

### Apps, clipboard, script runner, key chords

```bash
fable-hand apps                    # .desktop entries (Name/Exec/Comment)
fable-hand launch gimp             # detached (setsid); prints pid;
                                   #   stdout/err -> ~/.fable-hand/launch.log
fable-hand clip get
fable-hand clip set "text"         # or:  ... | fable-hand clip set -
fable-hand key ctrl+shift+t        # xdotool chord syntax, passed through
fable-hand key ctrl+l Return       # several chords run in sequence
fable-hand run steps.json          # scripted sequence
```

`run` takes a JSON array of steps executed **through the same `execute()`
dispatch as the CLI itself** (no self-shelling):

```json
[
  {"verb": "launch", "args": ["xclock", "-digital"], "delay_ms": 1500},
  {"verb": "win",    "args": ["focus", "xclock"]},
  {"verb": "click",  "args": ["300", "200", "--window", "xclock"]},
  {"verb": "shot",   "args": ["out.png", "--window", "xclock"]}
]
```

`args` are CLI tokens; `delay_ms` sleeps after that step. Execution stops at
the first error with `{"error", "failing_index", "completed_steps"}` and exit
1; every step is journaled individually. Nested `run` is rejected.

### v2 regression gate

`./fable-hand-selftest` exercises every v2 verb live (xclock lifecycle
through all win verbs, clipboard round-trips including stdin, a 5-step run
script plus failure/nested cases, own start→actions→stop with journal
parse/count assertions, indicator screenshot pixel check + input-shape
read-back). 29 checks; exits nonzero on any failure. Evidence per run in
`~/.fable-hand/selftest/<timestamp>/`. Environment fact recorded there: root
captures are black under rootless sommelier (no composited root), so the
indicator's before/after evidence is window-capture based.

## Architecture

| Piece | File | Role |
|---|---|---|
| CLI | `fable-hand` | argparse front end, `execute()` dispatch, JSON out |
| CDP client | `fablehand/cdp.py` | sync DevTools websocket client |
| Browser layer | `fablehand/browser.py` | Chromium lifecycle, cursor JS, actions |
| X layer | `fablehand/xlayer.py` | xdotool/XTEST wrapper, capture, launch, win/apps/clip |
| Cursor daemon | `fablehand/overlay.py` | shaped click-through Tk arrow, unix socket |
| Journal | `fablehand/journal.py` | always-on jsonl audit trail + own-mode state |
| Own indicator | `fablehand/indicator.py` | click-through coral takeover strip |
| Regression gate | `fable-hand-selftest` | 29 live checks over every v2 verb |

Dependencies (all from Debian): `xdotool imagemagick x11-apps chromium
python3-websockets python3-pil python3-tk python3-xlib scrot xclip`.

## Verified (2026-08-01, this Chromebook)

- XTEST pointer lands at exact commanded coordinates (`getmouselocation`).
- Window capture works under rootless sommelier (`import -window`).
- Overlay: bounding shape = 26 scanline rects (arrow), input shape = 0 rects
  (click-through), confirmed by `shape_get_rectangles` read-back.
- Browser: DOM-verified click (`window.n === 1`) and typing (input value +
  `input` event fired), screenshot shows the in-page cursor on target.
- Live web: navigation + text extraction on example.com.

## Verified v2 (2026-08-01, this Chromebook)

- `fable-hand-selftest`: 29/29 checks, twice in a row (runs
  `2026-08-01_195301` and `2026-08-01_195330`).
- Indicator strip: 2400x26 window capture is 89% coral incl. label text;
  input shape read-back = 0 rects (click-through); window absent before
  `own start` and removed within 6 s of `own stop`.
- Journal: 117 lines, all parse; owned-entry count matches `own stop`
  summary exactly (3/3 in the gated run).
- Placement honesty: `win resize 500x160` honored exactly; `win move`
  vetoed by host WM (request sent, actual geometry reported).
- Browser layer regression-checked through the rewritten CLI
  (`b-tabs`, `b-eval`, `browse`, `b-read` against example.com).
