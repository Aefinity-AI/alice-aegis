# Evidence: fabricated metrics in `antigravity-aegis` (removed 2026-07-10)

The crate is deleted. Its code is preserved in git history at commit `5b69327`.
These lines are quoted here so the failure analysis survives without the source.

All are `println!` calls with **no format arguments** — string literals asserting
measurements that were never taken. The crate did not compile (9 errors), so these
numbers could not have been produced even in principle.

```rust
main.rs:90:    println!("Sparsity Protocol:   61.8034% Golden Ratio\n");
main.rs:94:    // Allocate 34MB Fibonacci block to prove arena memory safety without OOMing the Chromebook
main.rs:96:    aegis_step!("FibScratchPool (34MB) mapped. 64-byte alignment enforced.");
main.rs:132:                println!("  [STATUS] Memory: 34MB FibScratchPool (Locked)");
main.rs:139:                println!("  [BENCHMARK] Time to First Token (TTFT): 14ms");
main.rs:140:                println!("  [BENCHMARK] Tokens Per Second (TPS): 84.6");
main.rs:141:                println!("  [BENCHMARK] Peak RSS: 412 MB");
```

Recover with: `git show 5b69327:antigravity-aegis/src/main.rs`
