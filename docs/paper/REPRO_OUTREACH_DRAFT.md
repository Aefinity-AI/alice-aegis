DRAFT ONLY. Nothing here is sent by Claude; Justin sends or does not.

## (a) Message

Subject: Can you run a 15-minute reproduction of a bit-identity check?

Hello,

We are writing a system paper on CIS-1, a contract for integer inference that
prints the same bytes on the x86_64 and aarch64 hosts we have tested so far. We would like an outside
reader to run our reproduction kit and tell us what they see.

The kit checks identity artifacts only: two digests and one golden receipt.
It needs git, stable Rust and a C linker. It takes about 15 minutes plus build
time. It does not ask for timing, and please do not send any.

Kit: [KIT URL]

If every line matches, please open a "Reproduction report" issue. If any line
differs, please open a "Divergence report" issue. We treat a divergence as a
finding, and we will record it with credit to you. We list reproductions in
the paper only with your consent, as architecture, CPU model, OS and result.

Thank you for your time.

## (b) Candidate reproducers

Suggestions, Justin picks. All contact channels are public pages. No emails.

| # | Target | Public channel | Reason |
|---|---|---|---|
| 1 | CommitLLM maintainers (lambdaclass) | https://github.com/lambdaclass/CommitLLM/issues | Receipt and verifier work in the same area; they can judge our receipt check. |
| 2 | bitnet.cpp maintainers (microsoft/BitNet) | https://github.com/microsoft/BitNet/issues | Ternary CPU inference on x86 and ARM; relevant hosts and review of our ternary claims. |
| 3 | Authors of arXiv 2606.00279 (Cankaya) | https://arxiv.org/abs/2606.00279 | Bit-exact inference verification across GPUs; we cover CPU ISAs. |
| 4 | Authors of arXiv 2602.00182 (EigenAI) | https://arxiv.org/abs/2602.00182 | Bit-exact inference with signed receipts on fixed GPU families. |
| 5 | Djiby Diop, UEFI llm-baremetal | https://huggingface.co/djibydiop/llm-baremey/discussions | Closest bare-metal UEFI project; the slug matches the Hub API record fetched on 2026-10-27. |
| 6 | Unikraft community | https://github.com/unikraft/unikraft/issues | Unikernel builders who can judge the execution model. |
| 7 | DiFR authors (arXiv 2511.20621) | https://arxiv.org/abs/2511.20621 | Statistical inference verification; a contrast with exact replay. |
| 8 | A university systems lab (category) | Department page or lab GitHub org issues page | Fresh toolchain and hosts; a neutral reader. |
| 9 | Rust embedded community (category) | https://github.com/rust-embedded/wg/issues | Readers used to `no_std` builds and odd targets. |
| 10 | An ARM or Raspberry Pi user (category) | A public forum thread or GitHub issue they own | A real aarch64 host outside our CI runners. |
