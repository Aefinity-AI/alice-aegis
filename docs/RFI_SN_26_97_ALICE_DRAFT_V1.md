# COVER PAGE
**Response to DARPA-SN-26-97 (Low Resource Computing)**
**Submission Date:** 16 JUL 2026

**Project Title:** A.L.I.C.E. (Aegis Lightweight Inference Core Engine) — Sovereign Bare-Metal AI for DDIL Operations
**Categories Addressed:** 
- Section 1(b): Low Memory
- Section 2(d): Low Complexity of User Experience

**Organization:** Ranger T / The ALICE Forge
**Technical POC:** Ranger T (email, address, phone)
**Admin POC:** Ranger T (email, address, phone)
**Proprietary Information:** None. 

---

# TECHNICAL SECTION (Max 8 Pages)

## Thesis Statement
In Denied, Degraded, Intermittent, and Limited (DDIL) environments, US forces require localized, autonomous intelligence that operates below the threshold of traditional power and networking grids. A.L.I.C.E. is a bare-metal, OS-less Large Language Model (LLM) unikernel designed to run on scavenged, low-tier legacy hardware (circa 2013-present x86 processors). By utilizing 2-bit ternary arithmetic and eliminating the operating system entirely, ALICE establishes an autonomous rear-security overwatch capability that can ingest raw sensor data and silently signal operators by hijacking ubiquitous Commercial Off-The-Shelf (COTS) IoT infrastructure (e.g., smart bulbs painted with IR-pass filters for NOD visibility). ALICE demonstrates that the minimum compute floor for advanced AI reasoning is significantly lower than current paradigms assume.

## 1. Capabilities and Challenges Addressed (Respondent's Perspective)
The reality of a Denied, Degraded, Intermittent, and Limited (DDIL) environment is not a degraded Wi-Fi signal; it is complete operational isolation in hostile or unfamiliar terrain. When a small unit or isolated soldier is cut off, every watt of power and every RF emission becomes a potential targeting solution for the adversary. In these critical moments, an operator does not need a cloud-connected dashboard, and a 400W server rack requiring a secure uplink is worse than useless—it is a liability. 

The capability gap DARPA-SN-26-97 addresses is fundamentally a survival and decision-support problem. The modern soldier carries an immense cognitive load, and fatigue rapidly degrades the working memory, spatial reasoning, and judgment required to navigate, triage, and survive. A.L.I.C.E. directly addresses this challenge by providing a tireless, autonomous lifeline that does not require an external network, a dedicated power plant, or complex training to operate. 

Booting a full intelligence capability off a 960MB USB stick into a scavenged 2019 laptop in under five seconds fundamentally alters the DDIL calculus. Traditional operating systems (Windows, Linux) are vulnerable; they require patching, they emit telemetry, they have massive attack surfaces, and they can lock out the user exactly when they are needed most (e.g., TPM/BitLocker trips in austere conditions). By completely eliminating the OS layer, ALICE runs as a zero-emission, zero-persistence unikernel. It turns the very fact of isolation into an asymmetric advantage: an AI that the enemy cannot jam, cannot track, and cannot exploit, because there is no network stack to attack and no hard drive left behind.

## 2. Theoretical / Simulation Discussion
*(Forge Note: We will expand on the technical architecture here.)*
- **The Unikernel Paradigm:** Eliminating the OS (Linux/Windows) removes the "OS tax" on memory and compute. ALICE boots directly from UEFI firmware (Ring 0), controlling physical memory directly via a custom allocator. 
- **Ternary Arithmetic (BitNet b1.58):** Using {-1, 0, +1} weights allows us to replace power-hungry Floating-Point Multiply-Accumulate (MAC) operations with highly optimized AVX2 SIMD additions/subtractions. 
- **Legacy Silicon Execution:** Explicitly countering the RFI's exclusion of high-tier CPUs. We rely on legacy/commodity silicon, avoiding advanced-node or HBM supply-chain dependencies. 

## 3. Development Strategy & Metrics
*(Forge Note: Explain the scale-down roadmap.)*
- **Phase VI Distillation:** The roadmap to train a highly distilled "mission model" tailored specifically for rear-security overwatch and DDIL doctrine Q&A. 
- **Power Efficiency:** Moving from the current ~18W laptop footprint down to embedded x86 SoC targets. 
- **IoT Mesh Integration:** The protocol for interfacing with local Bluetooth/Zigbee networks to control COTS hardware for covert IR signaling.

## 4. Identification of Current Data (Empirical Metrics)
To demonstrate extreme backwards compatibility and capability on scavenged DDIL hardware, A.L.I.C.E. was benchmarked on a legacy 2015 laptop processor executing strictly off a USB flash drive.
- **Execution Environment:** Intel Core i5-5200U (Broadwell architecture, circa 2015), no OS.
- **Footprint:** 545.41 MB Peak Physical Memory (Arena High-Water Mark). Operating System footprint = 0 bytes.
- **Decode Throughput:** 0.61 tokens/second (single-threaded AVX2+FMA). 
- **Context Resilience:** 0.59 tokens/second at 400 tokens context depth (demonstrating near-zero degradation as context scales). 
- **Security Validation:** Host OS BitLocker TPM lock triggered upon return to Windows, proving zero-trace, fully air-gapped execution outside the host's purview.

## 5. Estimated Time to Availability & Risk Assessment
- **Time to Availability:** Core engine is operational today (TRL 4/5). Distilled mission-model and IoT mesh integration estimated at 6-9 months for TRL 6. 
- **Technical Risks:** Scaling to larger context windows without an OS paging file; mitigation via custom NVMe direct-access drivers. 

---

# REFERENCES
1. Microsoft Research: *The Era of 1-bit LLMs* (BitNet b1.58)
2. ALICE Technical Report & Bare-Metal Benchmarks (July 2026)

---

# SUMMARY SLIDE (Placeholder)
*(Forge Note: For the Wednesday PM block. Will contain a visual architecture strip showing USB -> UEFI Boot -> Ternary Engine -> Covert IoT signaling, with the 3 strongest numbers overlaid.)*
