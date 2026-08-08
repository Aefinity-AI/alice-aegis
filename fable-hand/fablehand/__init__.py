"""fable-hand: Fable 5's local computer-use + browser utility.

Layers:
  xlayer    - XTEST input, window capture/management, apps, clipboard
  browser   - Chromium via CDP with a visible in-page animated cursor
  overlay   - visible synthetic cursor window for the X layer
  indicator - own-mode "FABLE-HAND OWNS THIS BOX" click-through strip
  journal   - always-on crash-safe action journal + own-mode state
  cdp       - minimal Chrome DevTools Protocol client

Honest scope: this drives apps inside the Crostini container (which display
as normal ChromeOS windows) and the full web via its own Chromium. It cannot
inject into the ChromeOS host UI itself; no guest->host input path exists.
"""

__version__ = "0.2.0"
