# Lookup tables for `agent_trace`

Each file is a tab-separated `key<TAB>value` table passed with `--table`. The
file's exact bytes are hashed into the trace genesis, so a receipt is bound to
the table it was answered from: swap one row and the receipt stops verifying.

## Why there are numeric-key tables here

`demo.tsv` and `chain.tsv` use `P-`-prefixed part numbers, which read naturally
but are **not a good test fixture for a small model**. E23 round 3 measured
what BitNet-2B actually emits when the prompt ends with an open `LOOKUP(`:

| prompt asks about | model wrote | parses? |
|---|---|---|
| `P-403` | `PI(403)` | no — `(` is outside the key grammar |
| `P-402` | `402)` | yes, but misses a `P-`-keyed table |
| `BOLT`  | `BOOLT)` | yes, but misses an alphabetic table |
| `402`   | `402)` | yes, **and hits** `parts-numeric.tsv` |

So with `P-` keys the episode produced `tool=no-tool` with empty `in=`/`out=`,
and the tamper matrix never exercised the tool-input/tool-output half of the
claim. That is a property of the model's spelling, not of the scanner — the
scanner correctly refused a malformed key, and `toks receipt` was what made
the cause visible instead of guessable.

`parts-numeric.tsv` exists so a 2B-class model can name a key it can actually
spell, and the tamper matrix gets a step with a real `in=` and a real `out=`.
`parts-numeric-chain.tsv` additionally routes `401 -> 404` and `406 -> 403`,
so a tool result names the next key: that is what a multi-step episode needs
if a later step's argument is to come from an earlier step's tool output
rather than from the prompt.

The `P-` tables are kept as-is. Receipts already issued against them depend on
their exact bytes.
