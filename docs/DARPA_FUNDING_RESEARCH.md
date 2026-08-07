# DARPA Edge-AI Funding — Verified Research (2026-07-10)

> Deep-research harness: 104 agents, each factual claim adversarially verified (3-vote refutation).
> Findings are primary-source (darpa.mil, DSIP, SBIR.gov). Vote 3-0 = unanimous survival; 2-1 = one dissent.

## Bottom line

As of mid-2026, DARPA's most on-topic vehicle for a small entity is the SBIR/STTR mechanism, run under the "Department of War (DoW) 2026 BAA" with all submissions through the DoD SBIR/STTR Innovation Portal (DSIP, dodsbirsttr.mil); Phase I is typically ~$250K/6 months, Phase II ~$1.8M/24-36 months, plus a Direct-to-Phase-II path and Phase III commercialization. Two DARPA efforts touch "low-resource" and edge AI directly — the SBIR topic Low Resource Computing (DPA26BZ02-DV010, now closed) and the FALCON SBIR topic (DPA26BZ03-DV016, opens July 22 2026) — but LRC is explicitly about repurposing legacy hardware without new silicon (software on constrained existing assets), and FALCON is about ML+LLM fusion for statistical analysis, so neither is a clean match for a bare-metal tiny-TCB inference appliance. Critically, DARPA's office-wide BAAs demand "revolutionary" (not "evolutionary"/efficiency) advances, and its two most relevant assured-AI programs (CLARA, and the Low Resource Computing RFI) are software-/proof-focused and hardware-agnostic — meaning DARPA rewards novel capability and human-explainable verifiability far more than tokens/joule, SWaP, or a small attack surface per se. The honest disconfirming picture: DARPA's documented "assurance" is proof-theoretic verifiability, not a small trusted computing base, and no located program directly rewards air-gapped/energy-instrumented CPU inference as a headline metric. To map onto what DARPA documents, the artifact should reframe its measured properties as an enabling novel capability (e.g., a demonstrably-infeasible-elsewhere capability on constrained/legacy commodity hardware) rather than as an incremental efficiency win.

## Verified findings

### 1. [high, vote 3-0 (claims 0,1,2)]
DARPA's Low Resource Computing (LRC) is an SBIR topic (DPA26BZ02-DV010, administered by the Small Business Programs Office) whose explicit goal is to repurpose existing/legacy DoD hardware to add net-new capability under strict resource limits WITHOUT designing new chips — i.e. a software/repurposing interest on constrained commodity hardware.

*Evidence:* Primary DARPA page: 'develop commercially viable re-use of existing DoW assets in lieu of new hardware investments ... This is critically not creating new chips, new computing architectures, etc.' and 'repurpose existing hardware to add net-new features not thought possible today due to resource limitations.' Opportunity ID DPA26BZ02-DV010, contact SBIR_BAA@darpa.mil. Targets end-of-life systems or those with <25% of new-system resources (e.g., old radio detecting a drone signature).

Sources:
- https://www.darpa.mil/research/programs/low-resource-computing
- https://www.darpa.mil/research/opportunities/dpa26bz02-nv010

### 2. [high, vote 3-0 (claim 3)]
For LRC Phase I, success requires demonstrating a capability otherwise believed infeasible on the target hardware (or that would take >2 years to field), and the Government evaluates 'the realism of the case made' plus a comparison against alternative state-of-the-art modernization options.

*Evidence:* Primary: prototype must demonstrate capabilities 'otherwise believed to (1) not be feasible on that hardware or (2) ability to deliver on the hardware would take over 2 years of effort'; 'the Government will evaluate the realism of the case made'; Phase I final report requires 'comparison with alternative state-of-the-art modernization options.' Narrow, single-topic criterion (upcycling legacy hardware), not DARPA-wide.

Sources:
- https://www.darpa.mil/research/programs/low-resource-computing
- https://www.darpa.mil/research/opportunities/dpa26bz02-nv010

### 3. [high, vote 3-0 (claims 4,5)]
DARPA has a parallel RFI titled 'Low Resource Computing' (DARPA-SN-26-97), issued by the Multi X Office (MXO), seeking low-resource computing paradigms not yet used in microsystems, response deadline July 17 2026 — but as an RFI it gathers information and awards NO funds (next step is an invitation-only concept workshop).

*Evidence:* Primary: 'RFI: Low Resource Computing ... Solicitation Type: RFI (Request for Information)'; 'The MXO seeks information on low resource computing paradigms and processes not yet utilized in microsystems.' Published 2026-06-18. Corroborated by SAM.gov/Govly/HigherGov. Creates no funding obligation; distinct from BAA or SBIR/STTR.

Sources:
- https://www.darpa.mil/work-with-us/opportunities/darpa-sn-26-97

### 4. [high, vote 3-0 (claims 7,8,9,13,14)]
DARPA SBIR/STTR phases and dollar amounts: Phase I ~$250K over ~6 months (scientific/technical merit + feasibility); Phase II ~$1.8M over 24-36 months (max Cost Volume $1.8M, max 36 months incl. options; typical structure $1.0M 18-24mo Base + $0.8M 6-12mo Option); Phase III derives from/extends prior work and focuses on commercialization; Phase II Follow-on Enhancement can add up to $500K in 1:1 matching funds over 12 months.

*Evidence:* DARPA overview page verbatim on $250K/6mo, $1.8M/24-36mo, Phase III commercialization, and $500K 1:1 Follow-on Enhancement. Phase II Instructions (2024-12, unchanged in Jan-2026 version): 'value of a DARPA Phase II award is typically $1,800,000 ... maximum ... $1,800,000 and maximum duration of 36 months including the proposed Option(s)' and 'typical structure is an 18 to 24-month, $1,000,000 Base and a 6 to 12-month, $800,000 Option ... Alternative structures may be proposed.' SBIR/STTR reauthorized April 2026 through Sept 2031.

Sources:
- https://www.darpa.mil/work-with-us/communities/small-business/sbir-sttr-overview
- https://www.darpa.mil/sites/default/files/attachment/2024-12/DARPA_SBIR_STTR_Phase_II_Instructions_09042024.pdf

### 5. [high, vote 3-0 (claims 19,20)]
DARPA offers a Direct to Phase II (DP2) mechanism allowing a Phase II award with no prior Phase I; DARPA also sets topic-specific Phase I amounts that can modestly exceed the nominal cap (e.g., 22.4 topic HR0011SB20224-08: $256K / 10 months / 25-page limit; DP2 topic HR0011SB20224-04: $1.5M / 24 months / 65-page limit).

*Evidence:* Primary 22.4 BAA: 'the Direct to Phase II (DP2) authority allows the DoD to make an award ... under Phase II ... without regard to whether the small business concern was provided an award under Phase I.' Award table lists HR0011SB20224-08 at $256,000/10 months and HR0011SB20224-04 DP2 at $1,500,000/24 months. Note: $256K is only ~$6K over the $250K guideline (modest, not 'substantial'); DP2 pilot dates to the 2011 reauthorization with DARPA as first implementer.

Sources:
- https://rt.cto.mil/wp-content/uploads/DARPA_SBIR_224_R5.pdf

### 6. [high, vote 3-0 (claims 10,11,18)]
All DARPA SBIR/STTR submissions go through the Defense/DoW SBIR/STTR Innovation Portal (DSIP, dodsbirsttr.mil) — there is no DARPA-specific portal; proposals by any other means are disregarded. Topics fall under the DoW 2026 BAA and are pre-released the first Wednesday of each month, during which (and only then) applicants may contact topic authors.

*Evidence:* DARPA: 'DOW pre-releases SBIR and STTR topics the first Wednesday of every month ... small businesses can view the topics and discuss technical questions directly with the topic authors ... Once the Announcement is open, direct questions ... are no longer allowed.' 'All submissions processed through: dodsbirsttr.mil.' BAA: 'Proposers are required to submit proposals via DSIP; proposals submitted by any other means will be disregarded.' 2026 branding sometimes 'DoW SBIR/STTR Innovation Portal' but mechanism unchanged.

Sources:
- https://www.darpa.mil/work-with-us/communities/small-business/sbir-sttr-overview
- https://www.darpa.mil/work-with-us/communities/small-business/sbir-sttr-topics
- https://rt.cto.mil/wp-content/uploads/DARPA_SBIR_224_R5.pdf

### 7. [high, vote 3-0 (claim 15)]
Eligibility for a solo/small entity: must be a for-profit U.S. business with 500 or fewer employees, more than 50% directly owned and controlled by U.S. citizens or permanent residents (the VC-majority-ownership pathway is agency-optional and NOT elected by DoD/DARPA).

*Evidence:* CRS R43695 and 13 CFR §121.702 / sbir.gov: for-profit U.S. business, 500 or fewer employees, 'more than 50% directly owned and controlled' by U.S. citizens or permanent resident aliens. 2026 sbir.gov eligibility tutorial confirms same thresholds.

Sources:
- https://www.congress.gov/crs-product/R43695
- https://www.sbir.gov

### 8. [high, vote 3-0 (claim 16)]
The SBIR/STTR program is a funnel — Phase I awards vastly outnumber Phase II (151,427 Phase I vs 68,077 Phase II historically, ~2.2:1) — reflecting many small feasibility awards and fewer advancing.

*Evidence:* sbir.gov/awards facet counts verbatim: Phase I = 151,427, Phase II = 68,077. Consistent with the documented cross-agency pattern where only ~50-60% of Phase I awardees who apply advance to Phase II.

Sources:
- https://www.sbir.gov/awards

### 9. [high, vote 3-0 (claim 21)]
DARPA evaluates each conforming SBIR proposal individually on its own merit (NOT competitively against other proposals); a proposal is 'selectable' only when its strengths outweigh its weaknesses with no accumulated weaknesses requiring extensive negotiation or resubmission (selectable ≠ guaranteed award; still subject to funding).

*Evidence:* BAA verbatim: 'Proposals will not be evaluated against each other during the evaluation process, but rather evaluated on their own individual merit'; 'A selectable proposal is a proposal ... [where] the strengths of the overall proposal outweighs its weaknesses ... [with] no accumulated weaknesses that would require extensive negotiations and/or a resubmitted proposal.' Consistent across 2022-2025 cycles.

Sources:
- https://rt.cto.mil/wp-content/uploads/DARPA_SBIR_224_R5.pdf

### 10. [high, vote 3-0 (claim 17)]
DARPA's office-wide BAAs (e.g., Strategic Technology Office HR001125S0001) explicitly seek REVOLUTIONARY advances and specifically exclude research yielding only evolutionary improvements to existing practice — so an edge-AI artifact must claim a novel capability, not merely better tokens/joule or efficiency.

*Evidence:* 'Proposed research should investigate innovative approaches that enable revolutionary advances ... Specifically excluded is research that primarily results in evolutionary improvements to the existing state of practice.' Caveat: this is DARPA-wide boilerplate (identical across DSO/TTO/I2O/BTO BAAs, dating back 15+ years), not distinctive STO/edge-AI guidance; the 'tokens/joule = evolutionary' reading is the researcher's defensible inference, and a large efficiency gain that unlocks a new capability class could itself qualify as revolutionary.

Sources:
- https://defencescienceinstitute.com/wp-content/uploads/2024/11/HR001125S0001.pdf

### 11. [medium, vote 3-0 (claim 6)]
The I2O (now 'Information Processing Techniques Office') office-wide BAA HR001126S0001 is a general-purpose avenue for unsolicited, revolutionary research not covered by existing programs, open to any responsible source including small businesses/solo researchers.

*Evidence:* Primary page: seeks 'revolutionary research ideas for topics not being addressed by ongoing programs or other published solicitations'; standard eligibility 'All responsible sources ...' Confidence lowered because multiple secondary sources (EverGlade, DSI) report the I2O office-wide BAA was 'temporarily paused' effective 21 May 2026 for reorganization (abstract/proposal deadlines listed Nov 1 / Nov 30 2026) — status must be re-verified on SAM.gov before relying on it.

Sources:
- https://www.darpa.mil/work-with-us/opportunities/hr001126s0001

### 12. [medium, vote 2-1 (claim 12)]
The active SBIR topic FALCON (Fusion of Abstract Learning and Context-Optimized Neural-methods, DPA26BZ03-DV016) targets computationally efficient ML fused with LLMs for interactive statistical analysis in enterprise/battlefield contexts (opens July 22, closes Aug 19 2026; ~$1.5M award) — relevant to edge ML+LLM but NOT to tiny-TCB/air-gapped/low-SWaP CPU inference.

*Evidence:* Topics page verbatim on name, topic #, office, dates, and 'combine advanced machine learning ... computationally efficient in structured data with large language models ... for interactive statistical analysis ... in enterprise or battlefield.' Split 2-1 vote; internal DARPA metadata inconsistency (program page lists DPA26BZ04-DV016 / SBPO vs DPA26BZ03 / 'Information Processing Techniques', a defunct office name). Topical fit to the artifact is loose.

Sources:
- https://www.darpa.mil/work-with-us/communities/small-business/sbir-sttr-topics
- https://www.darpa.mil/research/programs/falcon

### 13. [high, vote 3-0 (claims 22,23)]
DARPA's assured-AI programs are software- and proof-focused, not hardware/SWaP/edge — CLARA (DARPA-PA-25-07-02) defines 'assurance' as human-explainable verifiability via automated logical proofs and vetted logic building blocks, with metrics like verifiability-without-loss-of-performance and polynomial time complexity. This is the key DISCONFIRMING evidence: DARPA's documented 'assurance' is proof-theoretic, NOT a small trusted computing base, and would not directly reward a bare-metal appliance's hardware properties, tokens/joule, latency, or attack-surface size.

*Evidence:* CLARA FAQ: 'CLARA's focus is on software rather than (directly) hardware aspects' and assurance 'means verifiability with strong explainability to humans, based on automated logical proofs and hierarchical, vetted logic building blocks.' TA1/TA2 are software (ML+AR composition, open-source library). No mention of SWaP, edge, joules, or on-device inference — a distinct evaluation axis from a small-attack-surface TCB.

Sources:
- https://www.darpa.mil/sites/default/files/attachment/2026-03/darpa-program-faq-clara.pdf
- https://www.darpa.mil/research/programs/clara
