# CIS-1 BitNet b1.58 2B-4T: userspace receipt, 3 x86 hosts (2026-10-16)

Chain steps (b) per-generation receipt and (c) host verifier reproduces it, on real hardware.
Harvested verbatim from `~/legs/alice-2b-<host>/` on w2, w3, w4 (scp, no edits). Not a benchmark; wall times are not quoted.

- alice-aegis pin: `aa0309d532cfacc749277e332bc9862b29db3f4b` (`cargo build --release --example cis_witness` in aegis-linux; `cis-verify` with `--features std`)
- Model artifacts (sha256 in each RESULT.txt): model `facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db`, embed `e32b99a25e345c65054f36dedf40329a89513cdf1a8195db64fc440fd364e077`, vocab `5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae`
- Prompt: "The quick brown fox jumps over the lazy dog near the old river bank." ; 8 new tokens
- Command: `job.sh` in this directory (the exact script; per-host copy is `<host>/run.sh`, differs only in NAME). Core calls:
  `cis_witness gen MODEL EMBED VOCAB 8 "<prompt>" > receipt-R{1,2}.txt` ; `cis_witness verify MODEL EMBED VOCAB receipt-R1.txt` ; `cis-verify receipt-R1.txt MODEL EMBED VOCAB`

| host | CPU | avx2 count in /proc/cpuinfo | kernel | R1 vs R2 | verify_witness_rc | verify_standalone_rc | receipt sha256 (first 16) |
|---|---|---|---|---|---|---|---|
| w2 | Intel Core i5-5200U @ 2.20GHz | 4 | 6.12.111 | SAME | 0 | 0 | a5103369549397bc |
| w3 | Intel Celeron N4020 @ 1.10GHz | 0 | 6.12.111 | SAME | 0 | 0 | a5103369549397bc |
| w4 | Intel Celeron N4020C @ 1.10GHz | 0 | 6.12.94 | SAME | 0 | 0 | a5103369549397bc |

Receipt sha256 `a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7` (see list below for each host's file).
Receipt contents: token-ids 578,7160,7437,304,279,9909,11,25146 ; cis-digest c404a984d548715b ; chain 7bfc1c74cbb8fe5b250fa5a0e2bd2a889ce2eea7394a08c91ea2c1255fbc7953.
`<host>/host.txt` = CPU model, avx2 count, kernel, captured over ssh at harvest time (not by the leg).

## sha256 of every file
    ed23f67152491c7545b9feb5df3a2d13e01968ff7eb5139c3add82014690f61e  job.sh
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w2/gen-R1.err
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w2/gen-R2.err
    82d50ed2df646ace9185b22d261394fb5e67105747c8a23cfe36faae35f49b01  w2/host.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w2/receipt-R1.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w2/receipt-R2.txt
    7bc6e5bd7561f18b526b71a13e25dc86eca85fb0fb736fe94c993fb452bff3f1  w2/RESULT.txt
    c761c446b06a95ccc52e135c4e5be26c3291ea9ff59883f75701e7f505a524d4  w2/run.sh
    9aa0b66aa8185306600a943224d856cfe7f1733afcd43b15755d10afb143d34a  w2/verify-standalone.log
    c7f1727e715ac19cac023e36dc45165acad0cf8378f132be540a3f5f1e163d5e  w2/verify-witness.log
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w3/gen-R1.err
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w3/gen-R2.err
    da51e0e272c06d5dc03d2c928244f52945e0022ec5052ef112f5aeffb36c85d7  w3/host.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w3/receipt-R1.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w3/receipt-R2.txt
    5757f84ab94cba631b47e65d2f31cfad18bc9e33e11d3432840f80c54c608bf4  w3/RESULT.txt
    7533b5c60c724baba855544a69904735cb8461864a58ff494f8756ccd90b4de5  w3/run.sh
    bf33b280d619a3e2720cd868527d87f0b3fcc9014e4ccf60937fa8f5bb53bcb2  w3/verify-standalone.log
    c7f1727e715ac19cac023e36dc45165acad0cf8378f132be540a3f5f1e163d5e  w3/verify-witness.log
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w4/gen-R1.err
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  w4/gen-R2.err
    5b19a3ff943b04894972906edf09fc76fc2c2637da2340ac7e3b201a327cf4cb  w4/host.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w4/receipt-R1.txt
    a5103369549397bcbb1f123ecae8414590fd9c33f2c02721e946c1aa91814cb7  w4/receipt-R2.txt
    c02aa0fe2d1a93114ebbd6e9de56bd5528e40341d1c918a756a5bf77ec0a2aa1  w4/RESULT.txt
    7d7c50f0a52ced0a7c7d9cda913d7b463e36e3229dc546f4dc1a2c73b2e84bd7  w4/run.sh
    f7c00f1a8ceb37fbe3ce5e18b556f7cdece84d6fe066206419cc5b1fe45a8750  w4/verify-standalone.log
    c7f1727e715ac19cac023e36dc45165acad0cf8378f132be540a3f5f1e163d5e  w4/verify-witness.log
