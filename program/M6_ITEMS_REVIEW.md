# M6 Domain Eval — 150 items for SME review (DRAFT, not yet frozen)

Review instructions: check each item's correct answer (marked with >>>) and
that no distractor is also defensible. Items marked [SME-FLAG] need your
judgment most. Reply with item ids to fix/replace (or 'approved'). After
approval the file is frozen + sha256-hashed into the ledger — no edits after.
Verification already applied: every item blind-solved by an independent model;
all 150 keys matched (135 workflow + 15 creeds), 0 ambiguity flags.


## edge_systems (50 items)

**edge_systems-000** (test): A field laptop with 2 GB RAM and no swap configured keeps locking up when processing large log files. Without repartitioning the disk, what is the quickest way to add swap space?
    A. Back up and reformat the /home partition as a dedicated swap partition
>>> B. Create a swap file with dd, then run mkswap and swapon on it
    C. Enable hibernation support in the firmware setup menu
    D. Increase the size reserved for /tmp in /etc/fstab
    _(A swap file needs no repartitioning and is active immediately with mkswap/swapon; reformatting /home destroys data, firmware hibernation settings do not create swap, and /tmp size has nothing to do with swap. — mkswap(8)/swapon(8) man pages)_

**edge_systems-001** (test): On a single-board computer with limited RAM and a slow SD card, what does enabling zram accomplish?
>>> A. It provides a compressed swap device backed by RAM instead of the SD card
    B. It transparently compresses every file written to the SD card
    C. It doubles the effective CPU clock for memory-bound workloads
    D. It mirrors the contents of RAM to the SD card for crash recovery
    _(zram creates a compressed block device in RAM commonly used as fast swap; it does not compress on-disk files, change CPU clocks, or mirror memory to storage. — kernel Documentation/admin-guide/blockdev/zram)_

**edge_systems-002** (test): A fanless embedded PC slows down after ten minutes of sustained load inside a hot vehicle. Which observation would confirm CPU thermal throttling?
    A. The kernel log shows repeated USB device disconnect events
    B. Free memory drops steadily while the workload runs
>>> C. CPU temperature nears its trip point while the reported clock frequency drops
    D. The disk queue length rises sharply during the workload
    _(Thermal throttling shows as high temperature coinciding with reduced clock frequency; USB errors, memory pressure, and disk queueing are unrelated failure signatures. — Linux sysfs thermal zone / cpufreq docs)_

**edge_systems-003** (dev): In the output of the Linux free command, what does the available column estimate?
    A. RAM that no process has touched since the system booted
    B. The total physical RAM installed in the machine
    C. Swap space that has not yet held any pages since boot
>>> D. Memory usable by new programs without swapping, including reclaimable cache
    _(MemAvailable estimates memory startable workloads can claim without swapping, counting reclaimable page cache; the other choices describe free RAM, total RAM, and swap, which are separate columns. — free(1) / /proc/meminfo MemAvailable)_

**edge_systems-004** (dev): What does lowering the Linux sysctl vm.swappiness value from 60 to 10 do?
>>> A. Makes the kernel less eager to move idle pages out to swap
    B. Shrinks the swap partition to one sixth of its former size
    C. Raises the scheduling priority of the swap daemon process
    D. Disables swap entirely until the next system reboot
    _(swappiness tunes the kernel's preference for reclaiming anonymous pages to swap versus dropping cache; it never resizes swap, there is no swap daemon priority, and only swapoff disables swap. — kernel Documentation/admin-guide/sysctl/vm.rst)_

**edge_systems-005** (test): A recovery laptop with 1 GB RAM struggles to run its desktop environment. Which command makes it boot to a text console by default to free memory?
    A. systemctl enable getty@tty1.service
>>> B. systemctl set-default multi-user.target
    C. systemctl isolate emergency.target
    D. systemctl mask multi-user.target
    _(set-default multi-user.target makes text-mode boot persistent; getty@tty1 is already enabled, isolate changes only the current session (and emergency mode is single-user), and masking multi-user.target breaks normal boot. — systemctl(1) set-default documentation)_

**edge_systems-006** (test): After an unattended field unit rebooted a service overnight, which kernel log line would indicate that the out-of-memory killer terminated a process?
>>> A. Out of memory: Killed process 1234 (analyzer)
    B. EXT4-fs error: htree_dirblock_to_tree bad entry
    C. CPU0: Core temperature above threshold, throttling
    D. usb 1-1: device descriptor read error -71
    _(The OOM killer logs an explicit Out of memory: Killed process line; the others indicate filesystem corruption, thermal throttling, and USB faults respectively. — kernel mm/oom_kill.c log format)_

**edge_systems-007** (test): A technician is partitioning a disk for a UEFI-only Linux installation. Which partition must be present for the firmware to load the bootloader?
    A. A BIOS boot partition of at least 1 MiB near the disk start
    B. A dedicated /boot partition formatted as ext2
>>> C. An EFI System Partition formatted with a FAT filesystem
    D. A hidden NTFS recovery partition at the start of the disk
    _(UEFI firmware reads bootloaders from a FAT-formatted EFI System Partition; a BIOS boot partition is only for legacy GRUB on GPT, and neither ext2 /boot nor NTFS recovery partitions are readable by generic UEFI firmware. — UEFI Spec 2.x, EFI System Partition)_

**edge_systems-008** (test): On a legacy BIOS machine, where does the firmware look for the first stage of the operating system boot code?
>>> A. In the first sector of the disk, the master boot record
    B. In a file named BOOTMGR in the root directory of drive C
    C. In the last sector of whichever partition is marked active
    D. In the fallback path of the EFI System Partition
    _(Legacy BIOS loads and executes the 512-byte MBR at sector zero; BOOTMGR is found later by Windows boot code, the active partition's boot sector is its first sector, and ESP paths are a UEFI concept. — BIOS/MBR boot sequence convention)_

**edge_systems-009** (dev): A GPT-partitioned disk must boot on an old legacy-BIOS-only PC using GRUB. What does GRUB need on that disk to embed its core image?
    A. A FAT32 partition labeled EFI at the start of the disk
    B. An extended partition containing a logical /boot volume
>>> C. A small unformatted BIOS boot partition flagged bios_grub
    D. A swap partition of at least 512 MiB before the root volume
    _(On GPT there is no post-MBR gap, so GRUB embeds core.img in a bios_grub partition; an ESP serves UEFI not BIOS, extended partitions are an MBR concept, and swap is irrelevant to boot. — GRUB manual, BIOS installation on GPT)_

**edge_systems-010** (dev): While troubleshooting a running Linux system, how can you tell it was booted in UEFI mode rather than legacy BIOS mode?
>>> A. The directory /sys/firmware/efi exists
    B. The output of lsblk shows the disk uses GPT
    C. The file /proc/cmdline contains quiet splash
    D. The dmesg output lists ACPI tables at startup
    _(/sys/firmware/efi is populated only on UEFI boots; GPT disks can boot in BIOS mode, and kernel command-line options and ACPI tables appear regardless of boot mode. — kernel EFI sysfs interface docs)_

**edge_systems-011** (dev): What does UEFI Secure Boot verify when it is enabled?
    A. That the disk is encrypted with a key sealed in the TPM
>>> B. That boot binaries are signed by a trusted key before they execute
    C. That the firmware supervisor password matches its stored hash
    D. That the installed operating system license is activated
    _(Secure Boot checks digital signatures on boot-chain binaries against enrolled keys; disk encryption, firmware passwords, and OS licensing are entirely separate mechanisms. — UEFI Spec, Secure Boot chapter)_

**edge_systems-012** (dev): A salvaged 4 TB drive shows only about 2 TB usable after partitioning on an old system. What is the most likely cause?
    A. The SATA controller negotiates only half of the drive's line rate
    B. The drive firmware holds back half the capacity as spare sectors
    C. The filesystem journal and inode tables consume the missing space
>>> D. The drive was partitioned with an MBR table, which tops out near 2 TiB
    _(MBR's 32-bit LBA fields cap addressable space at about 2 TiB with 512-byte sectors, so GPT is required for the full 4 TB; link speed affects throughput not capacity, spare-sector reserves are tiny, and journals use a trivial fraction of space. — MBR 32-bit LBA limit vs GPT)_

**edge_systems-013** (test): Copying a 6 GB disk image to a freshly formatted FAT32 USB stick fails partway even though the stick has 32 GB free. What is the cause?
    A. The USB port is falling back to USB 1.1 transfer speeds
>>> B. FAT32 cannot store any single file of 4 GiB or larger
    C. The stick's partition table is GPT instead of MBR
    D. The FAT32 root directory is limited to 512 entries
    _(FAT32's 32-bit file size field caps files at 4 GiB minus one byte; slow ports only delay copies, GPT vs MBR does not block file writes, and the 512-entry root limit belongs to FAT16, not FAT32. — Microsoft FAT32 spec, 32-bit file size field)_

**edge_systems-014** (test): On a FAT filesystem, what short 8.3 alias would typically be generated for the long filename CONFIGURATION.TXT?
    A. CONFIGUR.TXT
    B. CONF~1.TXT
    C. CONFIGT1.TXT
>>> D. CONFIG~1.TXT
    _(The standard convention keeps the first six characters and appends a tilde plus a number, giving CONFIG~1.TXT; plain truncation to eight characters omits the required tilde disambiguator and the other forms do not follow the algorithm. — FAT 8.3 short-name generation (MS spec))_

**edge_systems-015** (test): For an x86-64 UEFI system to boot from a USB stick with no stored boot entries, which file path does the firmware look for on the stick's FAT partition?
    A. /boot/grub/grub.cfg
    B. /EFI/BOOT/BOOTIA32.EFI
    C. /syslinux/syslinux.cfg
>>> D. /EFI/BOOT/BOOTX64.EFI
    _(The removable-media default path for 64-bit x86 firmware is \EFI\BOOT\BOOTX64.EFI; BOOTIA32.EFI is the 32-bit fallback name, and grub.cfg or syslinux.cfg are loader configs read only after a loader is already running. — UEFI Spec, removable media boot behavior)_

**edge_systems-016** (dev): A bootable utility USB must be bootable by nearly any UEFI firmware in the field. Which filesystem should its boot partition use?
    A. ext4
    B. NTFS
>>> C. FAT32
    D. exFAT
    _(The UEFI spec mandates firmware support for FAT variants, so FAT32 boots almost everywhere; ext4, NTFS, and exFAT drivers are not guaranteed to exist in firmware. — UEFI Spec, ESP filesystem requirement)_

**edge_systems-017** (test): When writing an ISO image to a USB stick with dd on Linux, which output target is correct?
>>> A. The whole device node such as /dev/sdb
    B. The first partition node such as /dev/sdb1
    C. A mounted directory such as /media/usb
    D. The device-mapper path such as /dev/dm-0
    _(Hybrid ISOs carry their own partition table so they must be written to the raw device; writing to a partition leaves the ISO's table misplaced, a mounted directory just creates a file, and dm-0 targets an unrelated mapped device. — hybrid ISO / dd imaging practice)_

**edge_systems-018** (test): After writing a hybrid Linux ISO to a 64 GB stick with dd, the stick reports only about 3 GB total capacity. What restores full capacity for reuse?
    A. Run fsck on the existing partition to repair its size fields
    B. Reformat the visible small partition in place as FAT32
>>> C. Wipe the partition table and repartition the whole device
    D. Use a vendor unlock tool to expose the hidden storage area
    _(The ISO wrote its own small partition layout, so wiping the table (wipefs/fdisk) and repartitioning recovers the full device; fsck cannot resize partitions, reformatting the small partition keeps its size, and no capacity is vendor-locked. — hybrid ISO layout / wipefs(8))_

**edge_systems-019** (dev): You have physical console access to a Linux box but no root password and no install media or network. Which boot-time approach lets you reset the password?
    A. Press Ctrl+Alt+F2 at the login prompt and run passwd from that terminal
    B. Boot normally, log in to the guest account, and escalate with sudo su -
>>> C. Edit the GRUB kernel line to add init=/bin/bash, remount / read-write, run passwd
    D. Choose the memtest entry in the GRUB menu and use its maintenance reset option
    _(Booting with init=/bin/bash gives an unauthenticated root shell from which passwd works after remounting root read-write; other virtual terminals still demand a password, guest accounts lack sudo rights, and memtest has no password facility. — GRUB kernel parameter password recovery procedure)_

**edge_systems-020** (test): A typo in /etc/fstab drops a server into emergency mode at boot. After logging in on the console, what must be done before the file can be saved with corrections?
>>> A. Remount the root filesystem read-write with mount -o remount,rw /
    B. Run systemctl daemon-reload so systemd rereads its mount units
    C. Bring up the loopback network interface with the ip command
    D. Run mkfs on the partition named in the faulty fstab entry
    _(Emergency mode leaves root mounted read-only, so it must be remounted read-write before editing; daemon-reload does not change mount state, loopback is irrelevant, and mkfs would destroy data. — systemd emergency mode / mount(8))_

**edge_systems-021** (dev): To repair a broken bootloader from a live USB, a tech mounts the installed root filesystem at /mnt. Which additional step is required before chroot /mnt so that grub-install works properly?
    A. Copy the live system's kernel image into /mnt/boot
>>> B. Bind-mount /dev, /proc, and /sys into the /mnt tree
    C. Set the root password inside the mounted system first
    D. Enable a network interface so packages can be fetched
    _(grub-install inside the chroot needs device nodes and kernel interfaces, provided by bind-mounting /dev, /proc, and /sys; copying kernels, setting passwords, or networking are not prerequisites. — chroot rescue procedure (distro rescue docs))_

**edge_systems-022** (test): Before running e2fsck to repair an ext4 filesystem, what state must that filesystem be in?
    A. Mounted read-write so the checker can lock open files
    B. Mounted with the sync option to flush pending writes
>>> C. Unmounted, or mounted read-only at most
    D. Backed by at least 4 GB of free swap space
    _(Running e2fsck on a filesystem mounted read-write risks severe corruption, so it must be unmounted or read-only; the mount options and swap size choices are irrelevant to safe checking. — e2fsck(8) man page warning)_

**edge_systems-023** (dev): The root filesystem on an offline server is 98 percent full. Which command best identifies which top-level directory is consuming the space?
    A. df -h /
    B. ls -lhS /
    C. find / -type d -empty
>>> D. du -xh --max-depth=1 /
    _(du with a depth limit sums actual usage per top-level directory on one filesystem; df only shows the total, ls sizes of directory entries are meaningless for contents, and finding empty directories locates no space. — du(1) man page)_

**edge_systems-024** (test): With no network access, a technician has a required .deb package on a USB stick. Which command installs it on a Debian-based system?
    A. apt-get download /media/usb/package.deb
    B. snap install /media/usb/package.deb
    C. tar -xzf /media/usb/package.deb -C /
>>> D. dpkg -i /media/usb/package.deb
    _(dpkg -i installs a local .deb directly with no network; apt-get download fetches from repositories, snap handles a different package format, and a .deb is an ar archive, not a gzipped tar. — dpkg(1) man page)_

**edge_systems-025** (dev): A field server crashed and rebooted overnight, and its systemd journal is persistent. Which command shows the log messages from the boot during which it crashed?
>>> A. journalctl -b -1
    B. journalctl -f
    C. dmesg --follow
    D. last -x reboot
    _(journalctl -b -1 selects the previous boot's records; -f and dmesg --follow tail the current boot only, and last shows reboot times without log content. — journalctl(1) man page)_

**edge_systems-026** (dev): Which command reads a drive's SMART health attributes to check for pending sector failures on an offline Linux system?
    A. hdparm -tT /dev/sda
    B. badblocks -sv /dev/sda
>>> C. smartctl -a /dev/sda
    D. fdisk -l /dev/sda
    _(smartctl queries the drive's SMART attribute table including pending/reallocated sector counts; hdparm benchmarks throughput, badblocks does a surface scan without reading SMART, and fdisk lists partitions. — smartctl(8) / smartmontools docs)_

**edge_systems-027** (test): A tech connects two computers' DE-9 serial ports with a straight-through cable for a file transfer, but neither side sees data. What does a proper null-modem connection change?
>>> A. It crosses the transmit and receive lines between the two ends
    B. It boosts the signaling voltage from 5 V up to 12 V
    C. It converts the electrical signaling from RS-232 to RS-485
    D. It adds a ferrite choke at each end to remove line noise
    _(Two DTE devices both transmit on the same pin, so a null modem crosses TX and RX (and typically handshake lines); voltage boosting, RS-485 conversion, and ferrites do not solve the pinout problem. — RS-232 null modem wiring convention)_

**edge_systems-028** (test): What are the classic default serial settings for connecting a terminal to a Cisco network switch console port?
    A. 115200 baud, 8 data bits, even parity, 2 stop bits
    B. 57600 baud, 7 data bits, odd parity, 1 stop bit
    C. 19200 baud, 8 data bits, no parity, 2 stop bits
>>> D. 9600 baud, 8 data bits, no parity, 1 stop bit
    _(Cisco console ports default to 9600 8N1 with no flow control; the other combinations mix wrong speeds, parity, or stop bits and will produce garbage or no output. — Cisco console port access guide)_

**edge_systems-029** (dev): Which kernel command-line parameter directs Linux boot messages and a login prompt to the first onboard serial port at 115200 baud?
    A. serial=COM1,115200
>>> B. console=ttyS0,115200
    C. output=ttyS0,115200
    D. console=tty1,115200
    _(console=ttyS0,115200 is the documented syntax for a serial console; serial= and output= are not kernel console parameters, and tty1 is the local video console rather than a serial port. — kernel Documentation serial-console.rst)_

**edge_systems-030** (test): A serial terminal session shows a stream of garbage characters instead of readable text. What is the most common cause?
    A. The cable shield is not grounded at either end of the run
>>> B. A baud rate mismatch between the terminal and the device
    C. The terminal emulator is missing UTF-8 character support
    D. The remote device has hardware flow control turned on
    _(Mismatched baud rates corrupt the framing of every byte, producing classic garbage output; grounding, UTF-8 support, and flow control issues cause noise, wrong glyphs, or stalls rather than uniformly garbled text. — UART framing / baud mismatch behavior)_

**edge_systems-031** (dev): A Raspberry Pi data logger keeps corrupting its SD card after weeks of continuous duty. Which change most reduces wear on the card?
    A. Formatting the card as NTFS instead of ext4
    B. Switching to a card with a higher speed class rating
>>> C. Redirecting high-frequency log writes to a tmpfs RAM disk
    D. Running fstrim from cron every five minutes
    _(Moving constant small writes into RAM eliminates most flash program/erase cycles; NTFS does not reduce writes, speed class affects throughput not endurance, and excessive fstrim adds controller activity without cutting writes. — embedded SD endurance practice (tmpfs/noatime))_

**edge_systems-032** (test): What does the wear-leveling logic inside an SD card or SSD controller do?
>>> A. Spreads write operations across all flash blocks so none wears out early
    B. Lowers the programming voltage gradually as the device ages
    C. Compresses incoming data so fewer bytes reach the flash cells
    D. Throttles write speed once the device passes half its rated life
    _(Wear leveling remaps logical blocks so program/erase cycles are distributed evenly across the flash; it does not adjust voltage with age, compress data, or throttle by life stage. — flash wear-leveling fundamentals (JEDEC))_

**edge_systems-033** (test): What is the purpose of issuing a TRIM (fstrim) operation to a solid-state drive?
    A. It rewrites fragmented files so future reads become sequential
>>> B. It tells the controller which blocks hold deleted data so they can be erased in advance
    C. It scans for weakening cells and remaps them into the spare area
    D. It moves rarely used data onto the slower half of the flash array
    _(TRIM informs the controller which LBAs the filesystem no longer uses, letting garbage collection pre-erase them and sustain write performance; it is not defragmentation, cell testing, or data tiering. — ATA TRIM / fstrim(8))_

**edge_systems-034** (test): For an embedded system that will rewrite its storage constantly for years, which NAND flash cell type offers the highest write endurance?
    A. QLC
    B. TLC
    C. MLC
>>> D. SLC
    _(SLC stores one bit per cell and tolerates far more program/erase cycles than MLC, TLC, or QLC, which trade endurance for density. — NAND SLC/MLC/TLC/QLC endurance ratings)_

**edge_systems-035** (test): Before deploying a binary optimized with AVX2 instructions, how can a tech confirm that the target Linux machine's CPU supports AVX2?
>>> A. Check for the avx2 flag in the /proc/cpuinfo flags line
    B. Run uname -m and confirm the output says x86_64
    C. Read the CPU MHz value reported in /proc/cpuinfo
    D. Look for sse2 mentioned in the dmesg boot messages
    _(The kernel exports supported ISA extensions as flags in /proc/cpuinfo, so grep for avx2; x86_64 architecture and clock speed do not imply AVX2, and sse2 is a different, much older extension. — /proc/cpuinfo flags field)_

**edge_systems-036** (test): A precompiled tool exits immediately with an Illegal instruction error on an older field laptop but runs fine on a newer one. What is the most likely reason?
    A. The binary was compiled for a big-endian architecture variant
    B. One of the older laptop's RAM modules is failing under load
>>> C. The binary uses CPU instructions the older processor does not implement
    D. The executable permission bit was stripped during the file copy
    _(SIGILL at startup on old hardware classically means the build targets newer ISA extensions such as AVX2; endianness mismatch would fail to execute at all on x86, bad RAM causes random crashes, and a missing execute bit gives Permission denied. — SIGILL / ISA extension mismatch)_

**edge_systems-037** (test): At runtime, what mechanism does x86 software use to discover which instruction-set extensions the processor supports?
    A. Reading the CPU model name string from the SMBIOS tables
>>> B. Executing the CPUID instruction and decoding its feature bits
    C. Timing a short benchmark loop and comparing the scores
    D. Querying the ACPI DSDT for processor capability objects
    _(CPUID returns feature-flag bits that enumerate supported extensions; SMBIOS strings are informational only, benchmarks measure speed not features, and the DSDT does not list ISA extensions. — Intel SDM, CPUID instruction)_

**edge_systems-038** (dev): A 12 V battery is connected across a 6 ohm resistive load. What current flows through the load?
    A. 0.5 A
    B. 6 A
>>> C. 2 A
    D. 72 A
    _(Ohm's law gives I = V/R = 12/6 = 2 A; the distractors come from inverting the ratio, copying the resistance, or multiplying instead of dividing. — Ohm's law fundamentals)_

**edge_systems-039** (dev): An indicator LED with a 2 V forward drop must run at 20 mA from a 12 V supply. What series resistor value is required?
    A. 100 ohms
>>> B. 500 ohms
    C. 600 ohms
    D. 250 ohms
    _(R = (12 - 2) V / 0.020 A = 500 ohms; 600 ohms forgets to subtract the LED drop, and the others result from decimal or subtraction errors. — LED series resistor calculation)_

**edge_systems-040** (test): A 12 V, 20 Ah battery powers a device drawing a steady 0.5 A. Limiting discharge to 50 percent of capacity, about how long will it run?
    A. 40 hours
    B. 10 hours
    C. 5 hours
>>> D. 20 hours
    _(Usable capacity is 20 x 0.5 = 10 Ah, and 10 Ah / 0.5 A = 20 hours; 40 hours ignores the depth-of-discharge limit and the others are arithmetic slips. — battery amp-hour capacity math)_

**edge_systems-041** (test): A device is rated at 5 V and draws 2 A at full load. How much power does it consume?
    A. 2.5 W
    B. 7 W
    C. 0.4 W
>>> D. 10 W
    _(P = V x I = 5 x 2 = 10 W; 2.5 W divides instead of multiplying, 7 W adds the numbers, and 0.4 W inverts the division. — power equation P = V x I)_

**edge_systems-042** (test): Which connector family is the widely adopted standard for interchangeable 12 V DC power connections among amateur radio emergency communications teams?
    A. XT60
>>> B. Anderson Powerpole
    C. 5.5 mm barrel jack
    D. Molex 4-pin
    _(Anderson Powerpole is the ARES/RACES-adopted genderless 12 V DC standard; XT60 dominates hobby RC, barrel jacks vary in polarity and size, and Molex 4-pin is a PC internal connector. — ARES/RACES Powerpole standardization)_

**edge_systems-043** (test): Two identical 12 V, 7 Ah sealed batteries are wired in series. What does the combination provide?
>>> A. 24 V at 7 Ah
    B. 12 V at 14 Ah
    C. 24 V at 14 Ah
    D. 12 V at 7 Ah
    _(Series connection adds voltages while amp-hour capacity stays that of one battery, giving 24 V at 7 Ah; doubling capacity requires parallel wiring, and doubling both would need four batteries. — series vs parallel battery connection rules)_

**edge_systems-044** (test): A 12 V feed over a long wire run drops too much voltage under load. Keeping the same length and load, what change reduces the drop?
    A. Move to wire with a higher AWG number
    B. Add an inline fuse rated closer to the load current
>>> C. Use thicker wire with a lower AWG number
    D. Twist the positive and negative conductors together
    _(Voltage drop is I x R, and thicker (lower AWG) wire has less resistance; higher AWG is thinner wire which worsens the drop, fuses add resistance, and twisting helps noise immunity not DC drop. — AWG gauge / voltage drop fundamentals)_

**edge_systems-045** (dev): Using the standard formula, approximately how long is a quarter-wave whip antenna cut for 146 MHz?
>>> A. About 19 inches
    B. About 76 inches
    C. About 38 inches
    D. About 6 inches
    _(234 / 146 MHz = 1.6 ft, about 19 inches; 38 inches is a half wave, 76 inches a full wave, and 6 inches is roughly a quarter wave for the 70 cm band instead. — ARRL quarter-wave formula 234/f(MHz))_

**edge_systems-046** (test): Two teams using 5 W handheld VHF radios on flat open terrain lose contact at about 8 miles. Which single change most improves the link?
    A. Increasing transmit power from 5 W to 8 W
    B. Switching the receivers to a narrower IF filter
    C. Shortening each antenna to half its current length
>>> D. Raising the antenna height at both ends
    _(VHF range on open terrain is limited by the radio horizon, so antenna height gains far more than a small power bump; a 2 dB power increase barely helps, narrower filters affect adjacent-channel rejection, and shortening antennas detunes them. — VHF line-of-sight / radio horizon propagation)_

**edge_systems-047** (test) [SME-FLAG]: Compared with VHF, why is UHF often preferred for radio work inside buildings and dense urban areas?
    A. UHF signals follow the ground and bend beyond the horizon
>>> B. UHF's shorter wavelengths couple through openings and reflect indoors more effectively
    C. UHF is immune to multipath fading between structures
    D. UHF carries much farther than VHF over open water paths
    _(Shorter UHF wavelengths pass through window and door apertures and exploit reflections better in structures, which is why public-safety in-building work favors UHF; ground-wave bending is an HF/low-band trait, UHF is not immune to multipath, and open-water reach is not the reason. — public-safety VHF vs UHF propagation guidance)_

**edge_systems-048** (test): An SWR meter placed at the transmitter reads 1.2:1 on the operating frequency. What does this indicate?
    A. The coax has an open circuit at the antenna end
>>> B. The antenna system is well matched to the feed line
    C. About half of the forward power is being reflected back
    D. The transmitter is producing strong spurious harmonics
    _(An SWR near 1:1 means the load impedance closely matches the line, with under 1 percent reflected power; an open feed line would read very high SWR, half-power reflection corresponds to roughly 5.8:1, and SWR meters do not measure harmonics. — SWR and impedance matching basics)_

**edge_systems-049** (test): A quarter-wave vertical element is needed for a 300 MHz link. Since a full wavelength at 300 MHz is 1 meter, how long should the element be, ignoring end effects?
    A. 50 cm
    B. 1 m
    C. 75 cm
>>> D. 25 cm
    _(A quarter of the 1 m wavelength is 25 cm; 50 cm is a half wave, 1 m a full wave, and 75 cm three quarters of a wave. — wavelength = c/f, quarter-wave arithmetic)_


## tactics_drills (40 items)

**tactics_drills-000** (dev): A rifleman in a moving squad suddenly receives effective enemy small-arms fire. According to the react-to-contact battle drill, what is his immediate action?
    A. Freeze in place and wait for a fire command from the squad leader
    B. Sprint forward alone and assault the enemy position immediately
>>> C. Take the nearest covered position and return well-aimed fire
    D. Throw a smoke grenade and move back toward the objective rally point
    _(The drill requires soldiers to seek cover and return fire immediately without waiting for orders, while freezing, lone assaults, and withdrawing all violate the drill's immediate actions. — TC 3-21.76 ch6 (battle drill: react to contact))_

**tactics_drills-001** (test): Immediately after returning fire in the react-to-contact drill, soldiers shout out three pieces of information about the enemy so teammates can locate the threat. Which three?
>>> A. Distance, direction, and description
    B. Size, activity, and location
    C. Azimuth, elevation, and rate of fire
    D. Speed, strength, and supporting weapons
    _(Soldiers announce the three Ds (distance, direction, description) to orient the element onto the enemy; size-activity-location is a SALUTE fragment and the other options are not part of the drill. — TC 3-21.76 ch6 (battle drill: react to contact))_

**tactics_drills-002** (test): During react to contact, once the team in contact is returning fire, the squad leader's first key judgment concerns which question?
    A. Whether the squad can reach the objective rally point before the enemy reacts
>>> B. Whether the squad can gain and maintain suppressive fire against the enemy
    C. Whether rucksacks should be cached before continuing the mission
    D. Whether the platoon sergeant has submitted an updated casualty report
    _(The squad leader must determine whether his element can achieve suppressive fire, which drives the decision to maneuver or break contact, while the distractors are administrative or movement matters irrelevant to that decision point. — TC 3-21.76 ch6 (battle drill: react to contact))_

**tactics_drills-003** (test): A squad in contact is ordered to break contact. How does the drill accomplish the disengagement?
    A. The whole squad turns and runs to the rear at the same moment
    B. The squad assaults through the enemy position and continues the mission
    C. The squad holds in place until darkness allows a silent withdrawal
>>> D. One element suppresses while another bounds away, repeating until contact is broken
    _(Break contact is fire and movement in reverse, alternating suppression and rearward bounds, whereas simultaneous flight, assaulting, or waiting for dark are contrary to the drill. — TC 3-21.76 ch6 (battle drill: break contact))_

**tactics_drills-004** (test): When ordering an element to break contact, the leader's disengagement order tells that element where to go. Which form does this instruction take?
>>> A. A direction and distance, a terrain feature, or a designated rally point
    B. A ten-digit grid to the enemy's suspected command post
    C. A detailed strip map covering the entire route back through friendly lines
    D. A count of rounds remaining in each soldier's basic load
    _(The break-contact order specifies movement by direction and distance, terrain feature, or rally point, and none of the other items are part of the disengagement instruction. — TC 3-21.76 ch6 (battle drill: break contact))_

**tactics_drills-005** (test): A patrol walks into the kill zone of a near ambush. What do the soldiers caught in the kill zone do?
    A. Take cover, hold their fire, and wait for the trail element to flank the enemy
    B. Drop rucksacks and run straight back along the route of march
>>> C. Return fire, throw grenades, and assault through the ambush position
    D. Form a hasty perimeter in the kill zone and request indirect fire
    _(Staying in a near-ambush kill zone is fatal, so soldiers in it immediately return fire, throw grenades, and assault through the ambush, while waiting, fleeing along the march route, or perimeter defense leaves them exposed. — TC 3-21.76 ch6 (battle drill: react to near ambush))_

**tactics_drills-006** (test): What criterion determines whether an ambush is classified as near or far for the purpose of selecting the correct battle drill?
    A. The time of day at which the ambush is initiated
    B. The size of the enemy force conducting the ambush relative to the patrol
    C. The number of casualties taken in the initial burst of fire
>>> D. Whether the enemy is within hand-grenade range of the kill zone
    _(Near versus far ambush is defined by hand-grenade range, not by time of day, relative force size, or casualties taken. — TC 3-21.76 ch6 (battle drills: react to ambush))_

**tactics_drills-007** (test): A patrol element is caught in the kill zone of a far ambush. What is the correct action for those soldiers?
    A. Throw grenades and assault straight across the open ground into the enemy
>>> B. Return fire from covered positions and suppress the enemy
    C. Cease firing and crawl backward until out of the kill zone
    D. Fix bayonets and wait for the order to charge on line
    _(In a far ambush the kill zone element returns fire and suppresses while another element maneuvers, because assaulting across open ground at that range is not feasible and the other options surrender fire superiority. — TC 3-21.76 ch6 (battle drill: react to far ambush))_

**tactics_drills-008** (test): During react to far ambush, what task is given to the element positioned outside the kill zone?
>>> A. Maneuver along a covered and concealed route to flank the enemy
    B. Rush into the kill zone to pull casualties out under fire
    C. Withdraw immediately to the objective rally point and prepare a hasty defense
    D. Mark the friendly flank with panels and remain stationary
    _(The element not in the kill zone maneuvers by a covered route to attack the ambush from the flank, while entering the kill zone, withdrawing, or standing fast fails to destroy the enemy. — TC 3-21.76 ch6 (battle drill: react to far ambush))_

**tactics_drills-009** (test): Which step comes first in the eight troop leading procedures?
    A. Issue the warning order
>>> B. Receive the mission
    C. Make a tentative plan
    D. Initiate movement
    _(Troop leading procedures begin with receive the mission; the warning order, tentative plan, and movement are later steps. — TC 3-21.76 ch2 (troop leading procedures))_

**tactics_drills-010** (test): Which activity is the eighth and final troop leading procedure, continuing throughout preparation and execution?
    A. Complete the plan
    B. Conduct reconnaissance
    C. Issue the operation order
>>> D. Supervise and refine
    _(Step 8 is supervise and refine, with inspections and rehearsals continuing throughout, while the distractors are steps 6, 5, and 7 respectively. — TC 3-21.76 ch2 (troop leading procedures))_

**tactics_drills-011** (test): A platoon leader has six hours until the operation begins. Applying the one-third/two-thirds rule, how should he budget planning time?
    A. Spend the first four hours planning and give the squads the last two hours to prepare
    B. Split the six hours evenly between his planning and squad rehearsals
>>> C. Spend about two hours on his own planning and leave four hours for subordinates
    D. Use all six hours perfecting the order and brief it during movement
    _(The leader uses no more than one third of available time (two of six hours) and leaves two thirds for subordinate preparation; every other allocation shortchanges subordinates. — TC 3-21.76 ch2 (troop leading procedures, time management))_

**tactics_drills-012** (test): A squad leader has only fragmentary information minutes after receiving a new mission. What should he do about the warning order?
>>> A. Issue it now with the information available and update it as details arrive
    B. Hold it until the complete operation order is finished
    C. Skip it and brief everything at the objective rally point
    D. Pass it only to the team leaders once the leader's reconnaissance is complete
    _(Warning orders are issued as soon as possible so subordinates can begin parallel preparation and are updated as information arrives, while delaying, skipping, or restricting the order wastes preparation time. — TC 3-21.76 ch2 (warning order / TLP step 2))_

**tactics_drills-013** (dev): Which sequence lists the five paragraphs of an operation order in the correct doctrinal order?
    A. Mission, Situation, Execution, Service Support, Communications
    B. Situation, Mission, Maneuver, Logistics, Command
    C. Enemy, Friendly, Mission, Execution, Signal
>>> D. Situation, Mission, Execution, Sustainment, Command and Signal
    _(The OPORD paragraphs are Situation, Mission, Execution, Sustainment, and Command and Signal, while the distractors reorder paragraphs or substitute nondoctrinal or obsolete headings. — TC 3-21.76 ch2 (combat orders / OPORD format))_

**tactics_drills-014** (test): In which operation order paragraph does the commander's intent appear?
    A. Paragraph 1, Situation
    B. Paragraph 2, Mission
>>> C. Paragraph 3, Execution
    D. Paragraph 5, Command and Signal
    _(Commander's intent opens paragraph 3, Execution, and is not part of the situation, the mission statement, or command and signal. — TC 3-21.76 ch2 (OPORD format))_

**tactics_drills-015** (dev): A Ranger student needs the current challenge and password before a patrol. Under which operation order paragraph is that information published?
    A. Paragraph 3, Execution, in the coordinating instructions
>>> B. Paragraph 5, Command and Signal
    C. Paragraph 1, Situation, under friendly forces
    D. Paragraph 4, Sustainment
    _(Challenge and password are signal items published in paragraph 5, Command and Signal, not in coordinating instructions, the situation paragraph, or sustainment. — TC 3-21.76 ch2 (OPORD paragraph 5))_

**tactics_drills-016** (dev): Which set of elements must a properly written mission statement in paragraph 2 of an operation order contain?
>>> A. Who, what, when, where, and why
    B. Task, conditions, and standards
    C. Size, activity, location, unit, time, and equipment
    D. Observation, avenues of approach, key terrain, obstacles, and cover
    _(The mission statement answers the five Ws, while task-conditions-standards is a training construct and the other options are the SALUTE and OAKOC memory aids. — TC 3-21.76 ch2 (mission statement))_

**tactics_drills-017** (test): The plan for casualty evacuation and the scheme for resupply of ammunition and water belong in which operation order paragraph?
    A. Paragraph 1, Situation
    B. Paragraph 2, Mission
>>> C. Paragraph 4, Sustainment
    D. Paragraph 5, Command and Signal
    _(Casualty evacuation and resupply are sustainment functions carried in paragraph 4, while the other paragraphs cover situation, the mission statement, and command and signal matters. — TC 3-21.76 ch2 (OPORD paragraph 4))_

**tactics_drills-018** (test): In the mission analysis memory aid METT-TC, what does the final letter C represent?
    A. Command and control
    B. Cover and concealment
    C. Counterattack options
>>> D. Civil considerations
    _(The C in METT-TC stands for civil considerations, while the distractors borrow terms from other doctrinal constructs. — TC 3-21.76 ch2 (METT-TC))_

**tactics_drills-019** (test): While analyzing troops and support available under METT-TC, which information is the squad leader weighing?
    A. The attitudes and likely reactions of civilians living near the objective area
>>> B. The number, readiness, and morale of his own soldiers and any attachments
    C. The military aspects of the terrain along the planned route
    D. The amount of daylight and total mission time remaining before execution
    _(Troops and support available covers friendly strength, readiness, and supporting assets, while civilians, terrain, and time map to the C, terrain-T, and time-T factors instead. — TC 3-21.76 ch2 (METT-TC))_

**tactics_drills-020** (dev): The memory aid OAKOC helps a leader analyze which factor of METT-TC?
    A. Enemy
    B. Mission
>>> C. Terrain and weather
    D. Time available
    _(OAKOC structures analysis of the military aspects of terrain under the terrain and weather factor, not enemy, mission, or time. — TC 3-21.76 ch2 (OAKOC terrain analysis))_

**tactics_drills-021** (test): Within OAKOC, which element describes ground whose seizure or control gives a marked advantage to whichever side holds it?
>>> A. Key terrain
    B. Obstacles
    C. Avenues of approach
    D. Cover and concealment
    _(Key terrain is defined by the marked advantage its control affords, while obstacles impede movement, avenues are routes, and cover and concealment protect from fire and observation. — TC 3-21.76 ch2 (OAKOC terrain analysis))_

**tactics_drills-022** (dev): A team leader observes six enemy soldiers digging fighting positions on a ridgeline. In a SALUTE report, under which element does digging fighting positions fall?
    A. Size
>>> B. Activity
    C. Equipment
    D. Location
    _(What the enemy is doing is reported under activity, while size covers the personnel count, location the place, and equipment the gear observed. — TC 3-21.76 (SALUTE report format))_

**tactics_drills-023** (test): Which expansion correctly spells out the SALUTE report format?
    A. Situation, Action, Location, Unit, Terrain, Enemy
    B. Strength, Area, Losses, Uniform, Time, Environment
    C. Size, Activity, Level, Unit, Terrain, Equipment
>>> D. Size, Activity, Location, Unit, Time, Equipment
    _(SALUTE stands for size, activity, location, unit, time, and equipment, and each distractor substitutes at least one incorrect word. — TC 3-21.76 (SALUTE report format))_

**tactics_drills-024** (test): Which priority of work is established immediately upon occupying a patrol base and is maintained continuously the entire time the base is occupied?
    A. Water resupply
>>> B. Security
    C. Weapons maintenance
    D. The rest and sleep plan
    _(Security is always the first priority of work and never lapses, while maintenance, water, and rest are conducted afterward by portions of the unit at a time. — TC 3-21.76 ch5 (patrol base priorities of work))_

**tactics_drills-025** (dev): Doctrinally, how long may a unit occupy a single patrol base?
>>> A. No more than 24 hours, except in an emergency
    B. No more than 6 hours under any circumstances
    C. Up to 72 hours if resupply is available
    D. Indefinitely, provided camouflage is maintained
    _(A patrol base is occupied no longer than 24 hours except in an emergency and is never reused, so the other durations contradict the standard. — TC 3-21.76 ch5 (patrol base))_

**tactics_drills-026** (dev): A squad leader is choosing a patrol base site from his map. Which location best fits the doctrinal selection criteria?
    A. A prominent hilltop clearing offering long-range observation in every direction
    B. A dry streambed offering easy, quiet foot movement through the area
    C. A harvested field on the edge of a village where water can be drawn from a well
>>> D. Dense vegetation in difficult terrain, off natural lines of drift
    _(Patrol bases are sited in difficult terrain and dense vegetation away from routes people naturally travel, while prominent hilltops, streambeds, and village edges violate the avoidance criteria. — TC 3-21.76 ch5 (patrol base site selection))_

**tactics_drills-027** (dev): During weapons maintenance in a patrol base, which rule governs how the work is performed?
    A. All weapons are disassembled at once immediately after occupation to save time
    B. Weapons are cleaned only after the patrol returns to friendly lines
>>> C. Only a portion of the weapons are broken down at any one time
    D. Only the leaders' weapons receive maintenance while in the field
    _(Maintenance is staggered so the patrol base never loses its ability to fight, while disassembling everything at once, deferring all cleaning, or limiting maintenance to leaders' weapons violates the priorities of work. — TC 3-21.76 ch5 (patrol base priorities of work))_

**tactics_drills-028** (test): Shortly after infiltrating, a patrol conducts a security halt and performs SLLS. What does SLLS stand for?
>>> A. Stop, look, listen, smell
    B. Secure, locate, load, signal
    C. Scan, lase, log, send
    D. Stop, lower, lock, stand by
    _(SLLS is stop, look, listen, smell, used to attune the patrol's senses to the environment, and the distractors are invented expansions. — TC 3-21.76 ch5 (security halts / SLLS))_

**tactics_drills-029** (dev): A squad moving in file makes a short security halt. What should each soldier do?
    A. Gather around the squad leader in the center to hear updated instructions
    B. Sit back to back in the middle of the trail to rest
    C. Remain standing and keep eyes on the soldier ahead
>>> D. Take a knee behind cover facing outward, covering an assigned sector
    _(At a security halt soldiers drop to covered positions facing outward with assigned alternating sectors to create all-around security, while clustering, resting, or facing inward abandons security. — TC 3-21.76 ch5 (security halts))_

**tactics_drills-030** (dev): Which formation is the basic fire team formation, normally used unless terrain or visibility dictates otherwise?
    A. The file
>>> B. The wedge
    C. The echelon right
    D. The line
    _(The wedge is the fire team's basic formation, and the file, echelon, and line are adopted only when terrain, visibility, or the tactical situation requires. — TC 3-21.76 ch4 (movement formations))_

**tactics_drills-031** (test): A fire team in wedge enters vegetation so thick the normal interval cannot be maintained. What does the team do?
    A. Halt in place and wait until a bypass around the vegetation can be found
    B. Shift into a line formation and push through abreast
>>> C. Collapse into a file, then reform the wedge when the terrain opens
    D. Double the interval between soldiers and continue in wedge
    _(When terrain closes in, the wedge collapses to a file and reforms once the terrain opens, while halting, going on line, or widening intervals are incorrect responses. — TC 3-21.76 ch4 (fire team wedge))_

**tactics_drills-032** (test): A platoon expects enemy contact at any moment as it approaches a suspected defensive position. Which movement technique is appropriate?
    A. Traveling
    B. Traveling overwatch
    C. Forced march
>>> D. Bounding overwatch
    _(Bounding overwatch is used when contact is expected because one element is always set to fire, while traveling and traveling overwatch suit situations where contact is not likely or only possible, and forced march is not a movement technique. — TC 3-21.76 ch4 (movement techniques))_

**tactics_drills-033** (dev): Under which condition does a unit use the traveling movement technique?
>>> A. Contact with the enemy is not likely and speed is desired
    B. Contact is expected within the next terrain feature
    C. The unit is crossing a large open danger area under enemy observation
    D. The unit is assaulting across the objective
    _(Traveling is used when contact is not likely and speed matters, whereas expected contact calls for bounding overwatch and danger areas and assaults are handled by other techniques and drills. — TC 3-21.76 ch4 (movement techniques))_

**tactics_drills-034** (test): Two elements are moving by bounding overwatch. Which statement describes the alternate bound method?
    A. Both elements move at the same time along parallel routes
    B. The rear element moves only up level with the overwatching element, never beyond it
>>> C. The rear element advances past the overwatching element to the next position
    D. The lead element moves back to the rear element after every bound
    _(In alternate bounds the trail element leapfrogs past the overwatch element, whereas moving only up level describes successive bounds and the other options are not overwatch methods. — TC 3-21.76 ch4 (bounding overwatch))_

**tactics_drills-035** (test): Which list gives the five principles of patrolling?
    A. Speed, surprise, violence of action, simplicity, security
>>> B. Planning, reconnaissance, security, control, common sense
    C. Camouflage, discipline, initiative, rehearsal, reporting
    D. Firepower, movement, communication, patience, stealth
    _(The five principles of patrolling are planning, reconnaissance, security, control, and common sense, while the distractors mix assault fundamentals and invented lists. — TC 3-21.76 ch5 (principles of patrolling))_

**tactics_drills-036** (test): A patrol leader uses all available information and good judgment to make sound, timely decisions during a patrol. Which principle of patrolling is he applying?
>>> A. Common sense
    B. Control
    C. Security
    D. Reconnaissance
    _(Common sense is defined as using all available information and good judgment to make sound, timely decisions, while control concerns command and discipline, security concerns protection, and reconnaissance concerns gaining information. — TC 3-21.76 ch5 (principles of patrolling))_

**tactics_drills-037** (test): A patrol leader is about to depart the objective rally point on a leader's reconnaissance. What does doctrine require him to issue before leaving?
    A. A complete fragmentary order to every soldier in the patrol
    B. A new running password and challenge for reentry to the objective area
    C. A written route overlay signed by the platoon sergeant
>>> D. A five-point contingency plan to the senior man staying behind
    _(Whenever a leader leaves the main body he issues a five-point contingency plan (GOTWA) to the senior man remaining, and a full FRAGO, new passwords, or signed overlays are not required by the procedure. — TC 3-21.76 ch5 (GOTWA / leader's reconnaissance))_

**tactics_drills-038** (test): In the five-point contingency plan GOTWA, what does the letter G cover?
    A. The grid coordinates of the objective
>>> B. Where the leader is going
    C. Guards posted while the leader is away
    D. The go or no-go criteria for the mission
    _(G states where the departing leader is going, and the plan does not use G for objective grids, guard posts, or abort criteria. — TC 3-21.76 ch5 (GOTWA))_

**tactics_drills-039** (dev): The final letter of the GOTWA contingency plan addresses which subject?
    A. Ammunition redistribution to be completed before the leader departs
    B. Alternate routes back to the friendly forward line
>>> C. Actions on contact for the leader's party and the main body
    D. Approval authority for any change to the plan
    _(The final A covers actions on contact for both the element leaving and the element staying behind, while ammunition, routes, and approval authority are not GOTWA elements. — TC 3-21.76 ch5 (GOTWA))_


## land_nav (25 items)

**land_nav-000** (dev): A map's declination diagram shows a G-M angle of 8 degrees easterly. A soldier measures a grid azimuth of 132 degrees on the map. What magnetic azimuth should he set on his compass to walk that line on the ground?
    A. 140 degrees
    B. 132 degrees
>>> C. 124 degrees
    D. 116 degrees
    _(With an easterly G-M angle you subtract the angle when converting grid to magnetic (132 - 8 = 124); 140 adds instead of subtracting, 132 skips the conversion, and 116 subtracts the angle twice. — TC 3-25.26, Direction (declination diagram / G-M angle conversion))_

**land_nav-001** (test): On a military map's declination diagram, the G-M angle is the angular difference between which two norths?
    A. The angle between true north and grid north
>>> B. The angle between grid north and magnetic north
    C. The angle between true north and magnetic north
    D. The annual change between old and updated magnetic north
    _(G-M stands for grid-magnetic, the angle between grid north and magnetic north used for azimuth conversion; true-to-grid is grid convergence, true-to-magnetic is magnetic declination, and annual change is a separate marginal note. — TC 3-25.26, Direction (declination diagram))_

**land_nav-002** (test): A compass sighting on a distant tower reads 300 degrees magnetic. The map's G-M angle is 6 degrees westerly. What grid azimuth should the soldier plot on the map?
>>> A. 294 degrees
    B. 306 degrees
    C. 300 degrees
    D. 288 degrees
    _(With a westerly G-M angle you subtract the angle when converting magnetic to grid (300 - 6 = 294); 306 adds instead, 300 skips the conversion, and 288 applies the angle twice. — TC 3-25.26, Direction (G-M angle conversion, westerly))_

**land_nav-003** (test): A soldier is following an azimuth of 250 degrees and needs to return along the same line to his start point. What back azimuth should he follow?
    A. 110 degrees
    B. 430 degrees
    C. 180 degrees
>>> D. 70 degrees
    _(Because 250 is greater than 180, subtract 180 to get the back azimuth (250 - 180 = 70); 430 adds 180 without staying inside 0-360, 110 subtracts from 360 instead, and 180 is just the rule's threshold value. — TC 3-25.26, Direction (back azimuth rule))_

**land_nav-004** (dev): A soldier's pace count is 65 paces per 100 meters. While navigating a level leg he counts 195 paces. About how far has he traveled?
    A. 100 meters
    B. 200 meters
>>> C. 300 meters
    D. 400 meters
    _(195 paces divided by 65 paces per 100 meters equals three 100-meter increments, or 300 meters; the other values would require pace counts of roughly 195, 98, or 49 per 100 meters. — TC 3-25.26, distance measurement by pace count)_

**land_nav-005** (test): Compared with level ground, how does climbing a steep slope usually change the number of paces a soldier needs to cover 100 meters?
>>> A. It increases, because steps get shorter
    B. It decreases, because steps get longer
    C. It stays the same if walking speed is held constant
    D. It drops to about half the level-ground count
    _(Steps shorten going uphill, so more paces are needed per 100 meters; steps lengthen going downhill (not uphill), speed does not fix stride length, and halving is the wrong direction entirely. — TC 3-25.26, pace count factors (slope, terrain, fatigue))_

**land_nav-006** (test): Which MGRS grid coordinate locates a point to the nearest 10 meters?
    A. A 4-digit coordinate
>>> B. An 8-digit coordinate
    C. A 6-digit coordinate
    D. A 2-digit coordinate
    _(Precision increases with digits: 4-digit places a point within 1,000 meters, 6-digit within 100 meters, and 8-digit within 10 meters, so only the 8-digit coordinate meets the requirement. — TC 3-25.26, Grids (MGRS coordinate precision))_

**land_nav-007** (test): When reading a grid coordinate from a military map, in which order are the easting and northing values read?
    A. Up first, then right
    B. Left first, then down
    C. Down first, then left
>>> D. Right first, then up
    _(The standing rule is read right, then up: easting is read first along the bottom, then northing up the side; the other orders reverse or invert the rule and produce transposed coordinates. — TC 3-25.26, Grids (read right, then up))_

**land_nav-008** (test): A soldier is unsure of his own location, but he can see a water tower and a hilltop that he can positively identify on his map. Which technique lets him fix his own position using azimuths to those two known points?
>>> A. Resection
    B. Intersection
    C. Dead reckoning
    D. Aiming off
    _(Resection locates your own unknown position from azimuths to two or more known features; intersection locates a distant unknown point from known positions, dead reckoning requires a known start point, and aiming off is a deliberate-offset steering technique. — TC 3-25.26, Direction (resection))_

**land_nav-009** (test): Two observation posts at known, plotted locations each shoot an azimuth to the same unidentified smoke column and plot the lines on a map; the point where the lines cross marks the smoke. What technique are they using?
    A. Resection
    B. Modified resection
>>> C. Intersection
    D. Dead reckoning
    _(Intersection locates an unknown distant point by sighting to it from two or more known positions; resection and modified resection solve the opposite problem of finding your own position, and dead reckoning tracks movement, not a remote target. — TC 3-25.26, Direction (intersection))_

**land_nav-010** (test): Which navigation method has the soldier steer by recognizable features such as ridgelines, streams, and road junctions, checking them against the map as he moves rather than holding a precise compass azimuth?
    A. Dead reckoning
>>> B. Terrain association
    C. Field-expedient direction finding
    D. Map orientation by inspection
    _(Terrain association means navigating by matching ground features to the map and adjusting en route; dead reckoning is the precise azimuth-and-distance method, and the other two are techniques for finding direction or orienting the map, not full navigation methods. — TC 3-25.26, Terrain Association)_

**land_nav-011** (test): Where is the contour interval of a standard military topographic map stated?
    A. On every contour line on the map
    B. Within the declination diagram box
    C. Printed next to each grid line number
>>> D. In the map's marginal information
    _(The contour interval note appears in the marginal information (near the bar scales); only index contours carry elevation labels, the declination diagram covers norths, and grid numbers label grid lines, not relief. — TC 3-25.26, marginal information / Elevation and Relief)_

**land_nav-012** (test): On a contour map, what does a series of closely spaced contour lines indicate about the ground?
>>> A. A steep slope
    B. A gentle slope
    C. A flat valley floor
    D. A high elevation
    _(Contours packed close together mean elevation changes quickly over a short distance, which is a steep slope; wide spacing indicates gentle or flat ground, and spacing by itself says nothing about absolute elevation. — TC 3-25.26, Elevation and Relief (slope from contour spacing))_

**land_nav-013** (dev): On a map, two closed contour circles sit side by side with lower ground between them, forming an hourglass shape. What terrain feature is the low ground between the two high points?
    A. A draw
    B. A depression
>>> C. A saddle
    D. A valley
    _(A saddle is the dip or low point between two areas of higher ground and appears as an hourglass between two closed circles; a draw is a small sloping drainage, a depression is a closed low with tick marks, and a valley is a larger stream-cut feature. — TC 3-25.26, Elevation and Relief (terrain features: saddle))_

**land_nav-014** (test): The U- or V-shaped contour lines that form a draw point in which direction?
    A. Toward lower ground, away from the hilltop
>>> B. Toward higher ground, up the slope
    C. Parallel to the nearest ridgeline
    D. Toward the closest stream junction
    _(In a draw the closed end of the U or V points uphill toward higher ground (upstream); contours pointing toward lower ground indicate a spur, and the other two options describe no doctrinal contour rule. — TC 3-25.26, Elevation and Relief (terrain features: draw vs spur))_

**land_nav-015** (test): A soldier standing on a terrain feature notices the ground slopes down in three directions and up in one. On which feature is he standing?
    A. A draw
    B. A saddle
    C. A valley
>>> D. A spur
    _(A spur juts out from higher ground, so from it the ground falls away in three directions and rises in one; a draw is the opposite (up in three, down in one), a saddle rises in two directions, and a valley has high ground on both sides. — TC 3-25.26, Elevation and Relief (terrain features: spur))_

**land_nav-016** (dev): On a military map, a closed contour line with short tick marks pointing toward its center represents what?
>>> A. A depression
    B. A hilltop
    C. A cut along a roadway
    D. A saddle
    _(A depression is shown as a closed contour with tick marks (hachures) pointing in toward the lower ground; a hilltop is a plain closed contour, a cut's hachures run along a road or railroad rather than a closed contour, and a saddle is an hourglass between two highs. — TC 3-25.26, Elevation and Relief (terrain features: depression))_

**land_nav-017** (dev): On a standard topographic map, which contour lines are drawn heavier and labeled with their elevation value?
    A. Every second line, the intermediate contours
    B. The supplementary contours shown as dashed lines
>>> C. Every fifth line, the index contours
    D. The two lines nearest each hilltop
    _(Index contours are every fifth contour, drawn bolder and labeled with elevation; intermediate contours are the lighter unlabeled lines between them, supplementary contours are dashed half-interval lines, and proximity to a hilltop confers no special treatment. — TC 3-25.26, Elevation and Relief (index/intermediate/supplementary contours))_

**land_nav-018** (dev): On a standard military topographic map, the color brown identifies which category of features?
    A. Vegetation such as woods and orchards
>>> B. Relief features such as contour lines
    C. Water features such as lakes and rivers
    D. Cultural features such as buildings
    _(Brown is used for relief features and elevation, including contour lines and cultivated land forms; vegetation is green, water is blue, and man-made cultural features are black (or red for major roads and built-up areas). — TC 3-25.26, topographic map symbols and colors)_

**land_nav-019** (dev): To measure the grid azimuth from Point A to Point B with a military coordinate scale and protractor, where does the soldier place the protractor's index point after drawing the line between the points?
    A. On Point B, with 0 pointed at Point A
    B. On the nearest grid line intersection
    C. Halfway between Point A and Point B
>>> D. On Point A, with 0 at the top of the map
    _(The index goes over the start point (Point A) with the 0/360 baseline aligned toward grid north at the top of the map, then the azimuth is read where the drawn line crosses the scale; placing it on Point B measures the back azimuth, and the other placements measure nothing meaningful. — TC 3-25.26, Direction (measuring azimuths with a protractor))_

**land_nav-020** (dev): An azimuth measured on the map with a protractor is which type of azimuth?
>>> A. A grid azimuth
    B. A magnetic azimuth
    C. A true azimuth
    D. A back azimuth
    _(A protractor aligned to the map's grid lines yields a grid azimuth, which must be converted with the G-M angle before use on a compass; magnetic azimuths come from the compass, true azimuths reference true north, and a back azimuth is any azimuth plus or minus 180. — TC 3-25.26, Direction (grid vs magnetic azimuth))_

**land_nav-021** (test): Which two pieces of information define each leg of a movement navigated by dead reckoning?
    A. Contour interval and map scale
    B. Elevation gain and rate of march
>>> C. An azimuth and a distance
    D. A handrail and an attack point
    _(Dead reckoning moves a measured distance along a set azimuth from a known start point, so direction and distance define each leg; contour interval and scale describe the map, elevation and rate of march are planning data, and handrails and attack points belong to terrain association. — TC 3-25.26, Dead Reckoning)_

**land_nav-022** (test): A squad moves cross-country while keeping a power line that parallels its route in sight on the flank, using it as a guide. In land navigation terms, the power line is being used as what?
    A. A catching feature
>>> B. A handrail
    C. An attack point
    D. A checkpoint
    _(A handrail is a linear feature running roughly parallel to the route that guides movement; a catching feature lies beyond the objective to signal overshoot, an attack point is a start point for the final precise leg, and a checkpoint confirms progress at a specific spot. — TC 3-25.26, Terrain Association (handrails))_

**land_nav-023** (test): A patrol notes that a river crosses its route about 200 meters beyond the objective; if the patrol reaches the river, it knows it has passed the objective. The river is serving as what?
    A. A handrail
    B. An attack point
    C. A steering mark
>>> D. A catching feature
    _(A catching feature (backstop) is a prominent feature beyond the objective that warns the navigator he has overshot; a handrail parallels the route, an attack point is near the objective for the final approach, and a steering mark is an object sighted along the azimuth ahead. — TC 3-25.26, Terrain Association (catching features))_

**land_nav-024** (test): A navigator plans to move quickly to an easily found road junction near a small, hard-to-spot objective, then follow a short, precise compass leg from the junction to the objective. The road junction is being used as what?
>>> A. An attack point
    B. A catching feature
    C. A handrail
    D. A rally point
    _(An attack point is an easily identifiable feature close to the objective from which a short, precise final leg is navigated; a catching feature lies past the objective, a handrail parallels the route, and a rally point is a patrolling control measure, not a navigation aid. — TC 3-25.26, Terrain Association (attack points); TC 3-21.76 land navigation)_


## cff_comms (20 items)

**cff_comms-000** (test): In a standard call for fire, the six elements are normally sent to the fire direction center in how many separate transmissions?
    A. One
>>> B. Three
    C. Six
    D. Nine
    _(Doctrine groups the six elements of a call for fire into three transmissions; 'one' and 'six' are common miscounts, and 'nine' conflates the call for fire with the 9-line MEDEVAC request. — ATP 3-09.30 (Observed Fires), call-for-fire format)_

**cff_comms-001** (test): Which two elements make up the first transmission of a standard call for fire?
>>> A. Observer identification and warning order
    B. Target location and full target description
    C. Warning order and method of engagement
    D. Target description and method of control
    _(The first transmission is observer identification plus the warning order; target location is the second transmission, and description, method of engagement, and method of fire and control form the third. — ATP 3-09.30, call-for-fire transmission sequence)_

**cff_comms-002** (test): An observer spots an enemy squad but doubts the target location is accurate enough for first-round effects. Which warning order applies?
    A. Fire for effect
    B. Immediate suppression
>>> C. Adjust fire
    D. At my command
    _(Adjust fire is used when the observer is unsure of the target location; fire for effect requires confidence in first-round accuracy, immediate suppression is for an immediate threat to friendlies, and 'at my command' is a method of control, not a warning order. — ATP 3-09.30, warning orders)_

**cff_comms-003** (dev): During adjustment of an area target, the observer requests fire for effect once an adjusting round is expected to impact within what distance of the target?
    A. 25 meters
>>> B. 50 meters
    C. 100 meters
    D. 200 meters
    _(The observer enters fire for effect when splitting the final 100-meter bracket places rounds within 50 meters of the target; the other distances misstate the bracketing standard. — TC 3-09.81 / ATP 3-09.30, adjustment and bracketing)_

**cff_comms-004** (dev): An adjusting round impacts short of the target and to the right of the observer-target line. Which subsequent correction is properly formed?
    A. Right 50, drop 100
    B. Add 100, left 50
>>> C. Left 50, add 100
    D. Drop 100, right 50
    _(A short round requires ADD and a round right of the OT line requires a LEFT correction, with deviation announced before range; the distractors reverse the direction of correction or the required sequence. — ATP 3-09.30, spotting and corrections)_

**cff_comms-005** (dev): An observer 4,000 meters from the target sees an adjusting round land 20 mils to the left of the target. Which deviation correction should be sent?
>>> A. Right 80
    B. Left 80
    C. Right 20
    D. Left 20
    _(The OT factor is 4 (4,000 m / 1,000), so 20 mils x 4 = 80 meters, and a round left of the OT line is corrected RIGHT; distractors either move the round further left or ignore the OT factor. — ATP 3-09.30, OT factor and mil relation)_

**cff_comms-006** (dev): For mortar and cannon artillery missions, the observer announces DANGER CLOSE when the target is within what distance of friendly troops?
    A. 400 meters
>>> B. 600 meters
    C. 750 meters
    D. 1,000 meters
    _(Danger close for mortars and field artillery is 600 meters; 750 meters applies to naval guns 5-inch and smaller, and the other values are not doctrinal thresholds for cannon artillery. — TC 3-09.81, danger close criteria)_

**cff_comms-007** (test): During a fire mission, the observer transmits REPEAT immediately after rounds impact. What is the observer asking the firing unit to do?
    A. Retransmit the last message word for word
>>> B. Fire again using the same firing data
    C. Return to the initial aim point of the mission
    D. Confirm receipt of the previous correction
    _(In fire support REPEAT means fire the same data again, which is exactly why REPEAT is never used on the radio to request retransmission; the distractors describe SAY AGAIN, a re-adjustment, and an acknowledgment. — ATP 3-09.30, proword REPEAT in fire missions)_

**cff_comms-008** (test): A radio operator missed the middle of an incoming message. Which proword requests retransmission of the missed portion?
    A. REPEAT
    B. READ BACK
    C. ROGER
>>> D. SAY AGAIN
    _(SAY AGAIN (with ALL AFTER / ALL BEFORE as needed) requests retransmission; REPEAT is reserved for fire missions, READ BACK directs the receiver to read a message back, and ROGER merely acknowledges receipt. — ATP 6-02.53, radiotelephone prowords)_

**cff_comms-009** (dev): Which proword tells the sender that a message was received, is understood, and will be complied with?
>>> A. WILCO
    B. ROGER
    C. OVER
    D. OUT
    _(WILCO means received, understood, and will comply; ROGER only acknowledges receipt, while OVER and OUT are transmission-ending prowords. — ATP 6-02.53, radiotelephone prowords)_

**cff_comms-010** (dev): Why is sending 'ROGER, WILCO' together considered improper radio procedure?
>>> A. WILCO already includes the meaning of ROGER
    B. ROGER already includes the meaning of WILCO
    C. WILCO may only be sent by the net control station
    D. ROGER may only be used to answer a radio check
    _(WILCO incorporates the acknowledgment meaning of ROGER, so the pair is redundant; the reverse is false, and neither proword is restricted to net control or radio checks. — ATP 6-02.53, radiotelephone prowords)_

**cff_comms-011** (test): A station ends a transmission and expects an immediate reply. Which proword should close the transmission?
    A. OUT
    B. BREAK
>>> C. OVER
    D. WILCO
    _(OVER means the transmission is finished and a response is expected; OUT means no reply is expected, BREAK separates portions of a message, and WILCO signals compliance rather than closing a transmission awaiting reply. — ATP 6-02.53, radiotelephone prowords)_

**cff_comms-012** (test): Using the NATO phonetic alphabet, which sequence correctly transmits the letters B, D, and J?
    A. Baker, Dog, Jig
    B. Bravo, David, Junior
    C. Beta, Delta, Julia
>>> D. Bravo, Delta, Juliett
    _(The NATO/ICAO alphabet uses Bravo, Delta, Juliett; 'Baker, Dog, Jig' is the obsolete WWII-era alphabet and the other options mix in non-standard words. — ATP 6-02.53 / ICAO phonetic alphabet)_

**cff_comms-013** (test): Which warning order is transmitted together with the target location in a single abbreviated transmission so fires can begin as quickly as possible?
    A. Adjust fire
    B. Fire for effect
>>> C. Immediate suppression
    D. Suppression of enemy air defenses
    _(Immediate suppression compresses observer identification, warning order, and target location into one transmission for speed; adjust fire and fire for effect use the full three-transmission format, and SEAD is a mission category rather than a warning order. — ATP 3-09.30, immediate suppression)_

**cff_comms-014** (test): In a 9-line MEDEVAC request, Line 1 reports which information?
    A. Radio frequency and call sign
    B. Number of patients by precedence
>>> C. Grid location of the pickup site
    D. Method of marking the pickup site
    _(Line 1 is the pickup site location by grid; frequency and call sign are Line 2, patients by precedence is Line 3, and site marking is Line 7. — ATP 4-02.2 (Medical Evacuation), 9-line MEDEVAC request)_

**cff_comms-015** (test): Line 4 of a 9-line MEDEVAC request covers special equipment. Which letter code requests a hoist?
    A. A
>>> B. B
    C. C
    D. D
    _(On Line 4 the codes are A-none, B-hoist, C-extraction equipment, D-ventilator, so B requests a hoist. — ATP 4-02.2, 9-line MEDEVAC request Line 4)_

**cff_comms-016** (test): Which item of information appears in a 9-line MEDEVAC request but is never an element of a call for fire?
    A. A grid coordinate
    B. The requesting station's call sign
    C. A brief description of what is at the location
>>> D. Number of patients by litter or ambulatory type
    _(Patients by litter or ambulatory type is Line 5 of the MEDEVAC request and has no call-for-fire counterpart; grids and call signs appear in both formats, and describing what is at the location is the call for fire's target description element. — ATP 4-02.2 Line 5 vs ATP 3-09.30 call-for-fire elements)_

**cff_comms-017** (dev): Which habit best reflects proper radio discipline on a tactical voice net?
    A. Repeating every message twice so no station misses it
    B. Spelling out full unit designations in the clear
    C. Keying the handset while deciding what to say
>>> D. Listening first, then sending a short planned message
    _(Radio discipline demands listening before transmitting and short, preplanned transmissions; routine double transmissions and clear-text unit names violate brevity and OPSEC, and keying while composing ties up the net. — ATP 6-02.53, radiotelephone discipline and brevity)_

**cff_comms-018** (test): An observer wants the firing unit to withhold fires until the observer personally gives the order to fire. Which method of control should be included in the call for fire?
>>> A. AT MY COMMAND
    B. WHEN READY
    C. CANNOT OBSERVE
    D. TIME ON TARGET
    _(AT MY COMMAND has the FDC report ready and fire only on the observer's command; WHEN READY is the default with no withholding, and CANNOT OBSERVE and TIME ON TARGET serve different control purposes. — ATP 3-09.30, method of fire and control)_

**cff_comms-019** (test): Using successive bracketing, an observer's first adjusting round is spotted OVER. The observer sends DROP 400 and the next round is spotted SHORT. Which range correction comes next?
    A. Drop 200
    B. Add 400
    C. Drop 100
>>> D. Add 200
    _(Successive bracketing halves the established 400-meter bracket and reverses direction after a SHORT spotting, giving ADD 200; the distractors either keep the wrong direction or halve incorrectly. — ATP 3-09.30, successive bracketing)_


## creeds (15 items)

**creeds-000** (test): Which United States military creed begins with the words 'Recognizing that I volunteered'?
>>> A. The Ranger Creed
    B. The Soldier's Creed
    C. The NCO Creed
    D. The Airman's Creed
    _(The Ranger Creed's first stanza opens 'Recognizing that I volunteered as a Ranger...'; the others open differently. — Ranger Creed, stanza 1)_

**creeds-001** (test): The Ranger Creed has a distinctive structure. Which statement describes it?
    A. Five paragraphs that mirror the five-paragraph OPORD
    B. Four lines shared with the Warrior Ethos
>>> C. Six stanzas whose first letters spell the word RANGER
    D. Three verses, one for each Ranger battalion
    _(The six stanzas begin R-A-N-G-E-R; the creed is unrelated to OPORD structure, the Warrior Ethos, or battalion count. — Ranger Creed structure)_

**creeds-002** (dev): The recitation of the Ranger Creed traditionally ends with which phrase?
    A. Follow me!
    B. Send me!
    C. This we'll defend!
>>> D. Rangers lead the way!
    _('Rangers lead the way!' closes the creed; 'Follow me' is the Infantry motto, 'This we'll defend' the Army motto, 'Send me' from Isaiah/SOF ethos. — Ranger Creed, closing)_

**creeds-003** (dev): Which of these lines is part of the Warrior Ethos?
    A. I serve the people of the United States
>>> B. I will never accept defeat
    C. I am an expert and I am a professional
    D. I stand ready to deploy, engage, and destroy
    _(The Warrior Ethos is: mission first, never accept defeat, never quit, never leave a fallen comrade. The other lines are from elsewhere in the Soldier's Creed. — Warrior Ethos / Soldier's Creed)_

**creeds-004** (test): Which creed opens with the sentence 'I am an American Soldier'?
    A. The NCO Creed
    B. The Rifleman's Creed
>>> C. The Soldier's Creed
    D. The Ranger Creed
    _(That is the Soldier's Creed's first line; the NCO Creed opens 'No one is more professional than I', the Rifleman's Creed 'This is my rifle'. — Soldier's Creed, line 1)_

**creeds-005** (test): Which list gives all four lines of the Warrior Ethos correctly?
    A. Mission first; never surrender; never quit; never leave a fallen comrade
    B. Mission first; never accept defeat; never rest; never leave a fallen comrade
    C. Duty first; never accept defeat; never quit; never leave a comrade behind
>>> D. Mission first; never accept defeat; never quit; never leave a fallen comrade
    _(The four lines are: 'I will always place the mission first. I will never accept defeat. I will never quit. I will never leave a fallen comrade.' Each distractor alters one line. — Warrior Ethos)_

**creeds-006** (test): Which creed opens with 'No one is more professional than I'?
>>> A. The NCO Creed
    B. The Soldier's Creed
    C. The Officer's Oath
    D. The Ranger Creed
    _(That is the opening of the Creed of the Noncommissioned Officer. — NCO Creed, opening)_

**creeds-007** (test): Which creed refers to its members as 'the backbone of the Army'?
    A. The Soldier's Creed
>>> B. The NCO Creed
    C. The Ranger Creed
    D. The Rifleman's Creed
    _(The NCO Creed states the NCO corps is known as 'the backbone of the Army'. — NCO Creed)_

**creeds-008** (test): 'This is my rifle. There are many like it, but this one is mine.' These are the opening lines of which creed?
    A. The Soldier's Creed
    B. The Infantryman's Creed
    C. The Sailor's Creed
>>> D. The Rifleman's Creed
    _(The Rifleman's Creed (US Marine Corps) opens with these lines. — Rifleman's Creed (USMC))_

**creeds-009** (test): In the Rifleman's Creed, complete the line: 'Without me, my rifle is useless. Without my rifle, ...'
>>> A. I am useless.
    B. the fight is lost.
    C. I cannot win.
    D. my mission fails.
    _(The creed continues 'Without my rifle, I am useless. I must fire my rifle true.' — Rifleman's Creed)_

**creeds-010** (dev): Which creed contains the lines 'I am a Warrior. I have answered my Nation's call'?
    A. The Sailor's Creed
    B. The Soldier's Creed
>>> C. The Airman's Creed
    D. The Ranger Creed
    _(The Airman's Creed opens 'I am an American Airman. I am a Warrior. I have answered my Nation's call.' — Airman's Creed, opening)_

**creeds-011** (test): The Airman's Creed line 'I am faithful to a Proud Heritage, a Tradition of Honor, ...' ends with which phrase?
    A. and a Culture of Excellence
>>> B. and a Legacy of Valor
    C. and a History of Victory
    D. and a Duty to Country
    _(The line concludes 'and a Legacy of Valor.' — Airman's Creed)_

**creeds-012** (test): Which creed opens with 'I am a United States Sailor'?
    A. The Seaman's Oath
    B. The Navy Hymn
    C. The Mariner's Creed
>>> D. The Sailor's Creed
    _(The Sailor's Creed begins with exactly that line; the other titles are not US Navy creeds. — Sailor's Creed, line 1)_

**creeds-013** (dev): The Sailor's Creed states: 'I proudly serve my country's Navy combat team with Honor, Courage and ...' Which word completes it?
    A. Integrity
    B. Duty
>>> C. Commitment
    D. Sacrifice
    _(Honor, Courage, Commitment are the Navy core values named in the creed. — Sailor's Creed / Navy core values)_

**creeds-014** (dev): The Ranger Creed stanza beginning 'Never shall I fail my comrades' pledges to shoulder more than my share of the task, 'one hundred percent...' — complete the phrase.
    A. and more besides.
>>> B. and then some.
    C. every single day.
    D. without complaint.
    _(The stanza ends '...whatever it may be, one-hundred-percent and then some.' — Ranger Creed, N stanza)_

