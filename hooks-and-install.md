# A.L.I.C.E. skill + hooks — Claude Code install

Three pieces, in order of importance. The skill teaches Claude; the hooks
enforce deterministically whether or not Claude remembers.

## 1. Install the skill

Personal (available in every project on your machine):

```bash
unzip alice-unikernel.skill -d ~/.claude/skills/
# verify — you must see SKILL.md at exactly this depth:
ls ~/.claude/skills/alice-unikernel/SKILL.md
```

Project-scoped instead (committed to the repo, so every clone gets it):

```bash
unzip alice-unikernel.skill -d /path/to/alice-repo/.claude/skills/
```

Claude Code loads the description at startup and pulls in the full skill when a
task touches the project; you can also invoke it directly with
`/alice-unikernel`. Skill directories are watched, so edits to SKILL.md take
effect in the current session.

## 2. Git pre-commit gate (the hard enforcement)

Runs the integrity tripwire on every commit and blocks on P0 findings —
fabricated metrics, unconditional verdicts, stride splits. No Claude in the
loop; it gates you and any collaborator too.

```bash
cd /path/to/alice-repo
mkdir -p tools
cp ~/.claude/skills/alice-unikernel/scripts/integrity_check.py tools/
git add tools/integrity_check.py
cp pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Note `.git/hooks/` is per-clone (git does not version it), so each machine
runs the `cp`/`chmod` step once. The `tools/` copy IS committed, which also
gives CI the same gate: `python3 tools/integrity_check.py .` as a pipeline
step, failing the build on nonzero exit.

## 3. Claude Code feedback hook (optional, immediate)

Gives Claude the tripwire report right after it edits a file, instead of at
commit time. Add to the repo's `.claude/settings.json` (create the file if
absent; if a `hooks` key already exists, merge this event into it):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$CLAUDE_PROJECT_DIR\" && out=$(python3 tools/integrity_check.py . 2>&1) || { printf '%s\n' \"$out\" >&2; exit 2; }"
          }
        ]
      }
    ]
  }
}
```

Semantics to know: PostToolUse fires after the edit already happened, so it
cannot undo anything — exit 2 here means "show the findings to Claude as
feedback," which with this skill installed prompts an immediate fix. The
pre-commit hook remains the actual gate. (If you ever write a *blocking* hook,
it must be PreToolUse with exit 2 — exit 1 blocks nothing in Claude Code.)

Verify registration inside a session with `/hooks`, and test the script the
way the docs suggest, by piping sample JSON:
`echo '{"tool_name":"Edit","tool_input":{"file_path":"x"}}' | sh -c '<command>'`.

## Smoke test (2 minutes)

1. `cd` into the repo, start `claude`.
2. Ask: "run the integrity tripwire on this repo" — the skill should trigger
   and run the script.
3. Ask: "what's the rule about EMBED.BIN's dtype here?" — it should answer
   from directive 3 without you pasting anything.
4. Make a junk commit adding `let x = 14.12; // mock PPL` to a live crate —
   the pre-commit hook must block it.
