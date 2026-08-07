#!/usr/bin/env python3
"""Pre-execution safety rail. Overdrive is not recklessness.

This exists because `skipDangerousModePermissionPrompt: true` is set in
settings.json — no confirmation dialog will appear before a destructive command.
The only thing between a typo and a wiped disk is this file.

The interesting rule is the disk one. This project legitimately writes raw images
to USB sticks with `dd`, dozens of times. So a blanket ban on `dd of=/dev/...`
would be worse than useless: it would be disabled within the hour. Instead the
guard reads `/sys/block/<dev>/removable` and permits writes to removable media
while blocking writes to fixed disks. Precision, not prohibition — a rule that
is always in the way gets turned off.
"""
import json
import os
import re
import sys


def is_removable(dev: str) -> bool:
    """dev like 'sda' or 'nvme0n1'. Fixed disks return False; unknown → False."""
    base = re.sub(r"\d+$", "", os.path.basename(dev))  # sda1 -> sda
    try:
        with open(f"/sys/block/{base}/removable") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def deny(reason: str):
    print(json.dumps({"decision": "deny", "reason": reason}))
    sys.exit(0)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # never let the guard itself break the session
    cmd = data.get("tool_input", {}).get("command", "")
    if not cmd:
        sys.exit(0)

    # Unconditional: no legitimate use in this project.
    HARD_DENY = [
        (r"rm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?!tmp|home/\w+/\.cache)",
         "rm -rf on a root path"),
        (r"rm\s+-rf\s+~(/)?(\s|$)", "rm -rf on the home directory"),
        (r"\bmkfs(\.\w+)?\s", "mkfs — reformat a filesystem"),
        (r"curl[^|]*\|\s*(ba|z|fi)?sh", "piping a downloaded script into a shell"),
        (r"wget[^|]*\|\s*(ba|z|fi)?sh", "piping a downloaded script into a shell"),
        (r"git\s+push\s+.*--force(?!-with-lease)", "git push --force (use --force-with-lease)"),
        (r"chmod\s+-R\s+777", "chmod -R 777"),
        (r":\(\)\{.*\|.*&.*\};:", "fork bomb"),
    ]
    for pat, why in HARD_DENY:
        if re.search(pat, cmd):
            deny(f"Blocked by guard hook: {why}. Explain the intent and propose a safer command.")

    # Raw block-device writes: allowed to removable media, blocked on fixed disks.
    for m in re.finditer(r"(?:of=|>\s*)(/dev/(?:sd[a-z]\d*|nvme\d+n\d+p?\d*|vd[a-z]\d*|mmcblk\d+p?\d*))", cmd):
        dev = m.group(1)
        if not is_removable(dev):
            deny(
                f"Blocked by guard hook: raw write to {dev}, which is NOT removable "
                f"(/sys/block/*/removable != 1). That is a fixed disk — likely this "
                f"machine's root. If you truly mean it, do it outside Claude."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
