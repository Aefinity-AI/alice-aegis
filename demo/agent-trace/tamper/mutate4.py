"""E23 round-4 tamper mutator.

Round 3 caught 360/360 mutants but only about half by a digest comparison;
the rest were parse-rejects, which prove the parser is strict rather than
that the hash chain binds. This mutator therefore partitions its output:

  W  well-formed by construction  -- the mutant is still a grammatically
     valid receipt, so the ONLY thing that can reject it is a digest or
     replay comparison. Every value it substitutes is a value some genuine
     receipt in the corpus actually carried, or a same-shape edit of one
     (nibble flip inside a 64-hex field, token-id bump inside a token list),
     and every structural edit renumbers the step labels and fixes K so the
     result is internally consistent.

  M  malformed on purpose -- keeps round 3's parser-strictness coverage
     (broken keywords, truncation, label desync, unknown fields, version
     bumps). Reported separately; it is NOT evidence that the chain binds.

Membership is not taken on trust: the harness re-checks every emitted mutant
with wellformed.py, an independent grammar checker, and any disagreement
between the class this file assigns and that checker's verdict is reported
as a defect rather than silently resolved.

usage: mutate4.py <target-receipt> <outdir> <corpus-receipt> [corpus-receipt ...]
prints: <n_well_formed> <n_malformed>
"""
import itertools, os, re, sys

ALT = 2                     # alternatives tried per transplant slot
HEX64 = re.compile(r'\A[0-9a-f]{64}\Z')
TOOLS = ('no-tool', 'calc', 'lookup', 'calc-error')
LEAD_KEYS = ("AEGIS-TRACE model embed vocab K N prompt-hex trace-chain "
             "table-sha256 suite-sha256 commit host").split()


def read(p):
    with open(p, 'r', encoding='utf-8', errors='surrogateescape') as fh:
        t = fh.read()
    ls = t.split('\n')
    if ls and ls[-1] == '':
        ls = ls[:-1]
    return ls


def lead_of(lines):
    d = {}
    for l in lines:
        if l.startswith('step ') or l.startswith('WARNING'):
            continue
        k, _, v = l.partition(' ')
        d.setdefault(k, v)
    return d


def step_fields(line):
    body = line.split(':', 1)[1].strip() if ':' in line else ''
    d = {}
    for f in body.split():
        k, s, v = f.partition('=')
        if s:
            d[k] = v
    return d


def set_step_field(line, key, val):
    return re.sub(rf'(?<=\b{re.escape(key)}=)\S*', val, line, count=1) \
        if f'{key}=' in line and step_fields(line)[key] != '' \
        else re.sub(rf'\b{re.escape(key)}=(\S*)', f'{key}={val}', line, count=1)


def relabel(steps):
    """Renumber a list of step lines to consecutive labels from 0."""
    return [re.sub(r'\Astep [0-9]+:', f'step {i}:', s) for i, s in enumerate(steps)]


def flip_nibble(s):
    c = s[-1]
    return s[:-1] + hex(int(c, 16) ^ 1)[2:]


def build(target, corpus_paths):
    lines = read(target)
    corpora = [read(p) for p in corpus_paths]
    muts = []                                   # (name, cls, bucket, lines)

    def add(name, cls, bucket, ls):
        muts.append((name, cls, bucket, list(ls)))

    step_idx = [i for i, l in enumerate(lines) if l.startswith('step ')]
    warn_idx = [i for i, l in enumerate(lines) if l.startswith('WARNING ')]
    lead = lead_of(lines)
    fmt = lead.get('AEGIS-TRACE', 'v2')

    # ---- value pools, drawn from every genuine receipt in the corpus ------
    pool64, pool_toks, pool_inout, pool_prompt, pool_commit, pool_host = (
        set(), set(), set(), set(), set(), set())
    for cl in corpora:
        for l in cl:
            if l.startswith('step '):
                f = step_fields(l)
                for k in ('decode-chain', 'ctx', 'q'):
                    if HEX64.fullmatch(f.get(k, '')):
                        pool64.add(f[k])
                if f.get('toks'):
                    pool_toks.add(f['toks'])
                for k in ('in', 'out'):
                    if k in f:
                        pool_inout.add(f[k])
            elif not l.startswith('WARNING'):
                k, _, v = l.partition(' ')
                if HEX64.fullmatch(v):
                    pool64.add(v)
                elif k == 'prompt-hex':
                    pool_prompt.add(v)
                elif k == 'commit':
                    pool_commit.add(v)
                elif k == 'host':
                    pool_host.add(v)
    pool_host |= {'aefinity-box2', 'penguin'}
    pool_commit |= {'0' * 40, 'a' * 40}

    def alts(pool, cur, n=ALT):
        return sorted(v for v in pool if v != cur)[:n]

    # ===================== W: well-formed by construction =================

    # W1 64-hex slot transplants and nibble flips. Both keep the field a
    #    64-lowercase-hex string, which is the whole of its grammar, so the
    #    receipt still parses and only a digest comparison can reject it.
    for i, line in enumerate(lines):
        if line.startswith('step '):
            for key in ('decode-chain', 'ctx', 'q'):
                cur = step_fields(line).get(key)
                if not cur or not HEX64.fullmatch(cur):
                    continue
                for j, v in enumerate(alts(pool64, cur)):
                    nl = list(lines); nl[i] = set_step_field(line, key, v)
                    add(f'W1.t.l{i}.{key}.{j}', f'wf:hex64-transplant/{key}', 'W', nl)
                nl = list(lines); nl[i] = set_step_field(line, key, flip_nibble(cur))
                add(f'W1.f.l{i}.{key}', f'wf:hex64-nibble/{key}', 'W', nl)
        else:
            k, _, cur = line.partition(' ')
            if k not in LEAD_KEYS or not HEX64.fullmatch(cur):
                continue
            for j, v in enumerate(alts(pool64, cur)):
                nl = list(lines); nl[i] = f'{k} {v}'
                add(f'W1.t.l{i}.{k}.{j}', f'wf:hex64-transplant/{k}', 'W', nl)
            nl = list(lines); nl[i] = f'{k} {flip_nibble(cur)}'
            add(f'W1.f.l{i}.{k}', f'wf:hex64-nibble/{k}', 'W', nl)

    # W2 header transplants: prompt-hex, commit, host, N. Each substituted
    #    value satisfies that field's grammar (a genuine prompt from another
    #    episode, 40 hex, a non-empty hostname, a positive integer strictly
    #    below the genuine N so the prompt+N position bound cannot be the
    #    thing that rejects it).
    for i, line in enumerate(lines):
        k, _, cur = line.partition(' ')
        if k == 'prompt-hex':
            for j, v in enumerate(alts(pool_prompt, cur)):
                nl = list(lines); nl[i] = f'{k} {v}'
                add(f'W2.prompt.{j}', 'wf:prompt-transplant', 'W', nl)
        elif k == 'commit':
            for j, v in enumerate(alts(pool_commit, cur)):
                nl = list(lines); nl[i] = f'{k} {v}'
                add(f'W2.commit.{j}', 'wf:commit-transplant', 'W', nl)
        elif k == 'host':
            for j, v in enumerate(alts(pool_host, cur)):
                nl = list(lines); nl[i] = f'{k} {v}'
                add(f'W2.host.{j}', 'wf:host-transplant', 'W', nl)
        elif k == 'N':
            for j, v in enumerate(str(x) for x in (8, 15) if str(x) != cur):
                nl = list(lines); nl[i] = f'{k} {v}'
                add(f'W2.N.{j}', 'wf:N-value', 'W', nl)

    # W3 step-line transplants: token list, tool enum, tool input/output.
    for i in step_idx:
        f = step_fields(lines[i])
        for j, v in enumerate(alts(pool_toks, f.get('toks', ''))):
            nl = list(lines); nl[i] = set_step_field(lines[i], 'toks', v)
            add(f'W3.toks.l{i}.{j}', 'wf:toks-transplant', 'W', nl)
        ids = f.get('toks', '').split(',')
        for pos in sorted({0, len(ids) // 2, len(ids) - 1}):
            if not ids[pos].isdigit():
                continue
            nids = list(ids); nids[pos] = str(int(nids[pos]) + 1)
            nl = list(lines); nl[i] = set_step_field(lines[i], 'toks', ','.join(nids))
            add(f'W3.tokbump.l{i}.{pos}', 'wf:token-id-bump', 'W', nl)
        for v in TOOLS:
            if v == f.get('tool'):
                continue
            nl = list(lines); nl[i] = set_step_field(lines[i], 'tool', v)
            add(f'W3.tool.l{i}.{v}', 'wf:tool-enum', 'W', nl)
        for key in ('in', 'out'):
            for j, v in enumerate(alts(pool_inout, f.get(key, ''))):
                nl = list(lines); nl[i] = set_step_field(lines[i], key, v)
                add(f'W3.{key}.l{i}.{j}', f'wf:tool-{key}-transplant', 'W', nl)

    # W4 cross-step field swaps (round 3's strongest class, kept verbatim in
    #    spirit: a field's value is exchanged between two steps, so every
    #    value in the file is still one this very receipt carried).
    for a, b in itertools.combinations(step_idx, 2):
        fa, fb = step_fields(lines[a]), step_fields(lines[b])
        for key in ('toks', 'decode-chain', 'ctx', 'q', 'in', 'out', 'tool'):
            if key not in fa or key not in fb or fa[key] == fb[key]:
                continue
            nl = list(lines)
            nl[a] = set_step_field(lines[a], key, fb[key])
            nl[b] = set_step_field(lines[b], key, fa[key])
            add(f'W4.{key}.l{a}.l{b}', f'crossstep-swap:{key}', 'W', nl)

    # W5 structural edits, renumbered and with K corrected, so the result is
    #    an internally consistent receipt describing a DIFFERENT episode.
    #    Round 3 made these edits without renumbering, which is why all 36
    #    of them were rejected by the step-label check instead of the chain.
    head = lines[:step_idx[0]] if step_idx else lines
    tail = lines[step_idx[-1] + 1:] if step_idx else []
    steps = [lines[i] for i in step_idx]

    def rebuild(newsteps, name, cls):
        if not newsteps:
            return
        h = [re.sub(r'\AK [0-9]+\Z', f'K {len(newsteps)}', l) for l in head]
        t = [l for l in tail
             if not (l.startswith('WARNING step ')
                     and int(l[len('WARNING step '):].split(':')[0]) >= len(newsteps))]
        add(name, cls, 'W', h + relabel(newsteps) + t)

    for a, b in itertools.combinations(range(len(steps)), 2):
        ns = list(steps); ns[a], ns[b] = ns[b], ns[a]
        rebuild(ns, f'W5.reorder.{a}.{b}', 'wf:reorder-renumbered')
    for i in range(len(steps)):
        rebuild(steps[:i] + steps[i + 1:], f'W5.drop.{i}', 'wf:drop-renumbered')
        rebuild(steps[:i + 1] + [steps[i]] + steps[i + 1:],
                f'W5.dup.{i}', 'wf:duplicate-renumbered')
    for r in range(1, len(steps)):
        rebuild(steps[r:] + steps[:r], f'W5.rot.{r}', 'wf:rotate-renumbered')

    # W6 WARNING-set edits. A WARNING line is optional and its shape is
    #    fixed, so moving, dropping or adding one leaves a valid receipt;
    #    the verifier must reject it by comparing against its own replay.
    for i in warn_idx:
        n = int(lines[i][len('WARNING step '):].split(':')[0])
        nl = list(lines); del nl[i]
        add(f'W6.drop.{n}', 'wf:warning-drop', 'W', nl)
        for m in range(len(step_idx)):
            if m == n:
                continue
            nl = list(lines)
            nl[i] = f'WARNING step {m}: tool argument not found verbatim in context'
            add(f'W6.move.{n}.{m}', 'wf:warning-move', 'W', nl)
    if not warn_idx and step_idx:
        for m in range(len(step_idx)):
            nl = list(lines)
            nl.insert(step_idx[-1] + 1,
                      f'WARNING step {m}: tool argument not found verbatim in context')
            add(f'W6.add.{m}', 'wf:warning-add', 'W', nl)

    # ===================== M: malformed on purpose ========================
    for i, line in enumerate(lines):
        head_tok = line.split(' ')[0]
        nl = list(lines); nl[i] = line.replace(head_tok, head_tok + 'X', 1)
        add(f'M.lead.l{i}', 'mf:lead-keyword', 'M', nl)
    for i in range(1, len(lines)):
        add(f'M.trunc.{i}', 'mf:truncate', 'M', lines[:i])
    for i in step_idx:
        n = int(lines[i][len('step '):].split(':')[0])
        nl = list(lines)
        nl[i] = re.sub(r'\Astep [0-9]+:', f'step {n + 1}:', lines[i])
        add(f'M.label.l{i}', 'mf:step-label', 'M', nl)
        nl = list(lines); nl[i] = lines[i] + ' extra=1'
        add(f'M.unkfield.l{i}', 'mf:unknown-field', 'M', nl)
        nl = list(lines); nl[i] = lines[i] + ' stray'
        add(f'M.stray.l{i}', 'mf:stray-token', 'M', nl)
    nl = list(lines); nl[0] = f'AEGIS-TRACE v{int(fmt[1:]) + 1}'
    add('M.version', 'mf:version', 'M', nl)
    for d in (1, -1):
        nl = [re.sub(r'\AK ([0-9]+)\Z', lambda m: f'K {int(m.group(1)) + d}', l)
              for l in lines]
        if nl != lines:
            add(f'M.kdesync.{d}', 'mf:K-desync', 'M', nl)
    for i in warn_idx:
        nl = list(lines); nl[i] = lines[i] + ' and also everything else'
        add(f'M.warn.l{i}', 'mf:warning-text', 'M', nl)
    if step_idx:
        nl = list(lines); nl.insert(step_idx[0], lines[0])
        add('M.dupmagic', 'mf:duplicate-line', 'M', nl)
    return muts


def main():
    target, outdir = sys.argv[1], sys.argv[2]
    corpus = sys.argv[3:] or [target]
    os.makedirs(outdir, exist_ok=True)
    base = '\n'.join(read(target)) + '\n'
    seen, nw, nm = {base}, 0, 0
    with open(os.path.join(outdir, 'INDEX.tsv'), 'w') as idx:
        for name, cls, bucket, ls in build(target, corpus):
            body = '\n'.join(ls) + '\n'
            if body in seen:          # never emit the identity or a duplicate
                continue
            seen.add(body)
            p = os.path.join(outdir, name + '.txt')
            with open(p, 'w', encoding='utf-8', errors='surrogateescape') as fh:
                fh.write(body)
            idx.write(f'{name}\t{cls}\t{bucket}\t{p}\n')
            nw += bucket == 'W'
            nm += bucket == 'M'
    print(f'{nw} {nm}')


if __name__ == '__main__':
    main()
