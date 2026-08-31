# GOP console screenshot (QEMU boot-test)
`gop-console-screenshot.png` — captured 2026-08-31 from `qemu-system-x86_64 -vga std` running the
qemu-test build (tinybit model). It documents that the GOP framebuffer console renders (DejaVu 16x32).
The `[PERFORMANCE] Average Cycles/Token` lines visible on-screen are TCG-emulation artifacts printed by
the test harness; per Rule A they are NOT performance figures and must never be quoted or compared.
