# Venue options for the CIS-1 system paper

Status: facts table, fetched 2026-10-27 from each venue's own pages (URLs in the
last column). The maintainer picks the venue; this file records facts and does
not recommend. Rows marked "no CFP posted" carry the previous cycle's dates as
a guide only. "AoE" = Anywhere on Earth.

The draft (`docs/paper/paper.md`) is 13,249 words of Markdown (`wc -w`,
2026-10-27), far longer than every page limit below at any typesetting
density, so any submission needs a cut-down version plus the full draft as an
appendix or public preprint.

## Candidate venues with a posted deadline after 2026-10-27

| Venue | Next deadline(s) | Page limit / format | Artifact evaluation | Preprint policy | Source (fetched 2026-10-27) |
|---|---|---|---|---|---|
| USENIX Security 2027, Denver, Aug 11-13 2027 (main conference) | Cycle 2: mandatory registration 2027-01-19; paper 2027-01-26; artifacts 2027-01-29; all AoE. Notification 2027-05-06. | 13 pages excluding references and appendices; camera-ready 20 pages total; USENIX template, double-blind. | Open Science Appendix required at submission (artifact links or a reason why not); artifact evaluation of accepted papers is optional; acceptance depends on public availability where artifacts can be public. | Authors may post preprints during review (arXiv not named on the page). | https://www.usenix.org/conference/usenixsecurity27/call-for-papers |
| USENIX SAIS '27 (Secure Agentic-AI Systems), Santa Clara, Jun 7-8 2027 (workshop-scale symposium) | Paper 2027-02-04 23:59 AoE; notification 2027-03-16; final 2027-04-20. No separate abstract deadline. | 13 pages excluding references and appendices; 30 pages total; USENIX template, double-blind. | Open Science Appendix required; anonymous artifact link at submission; stable non-anonymous link at camera-ready. No separate AE track found on the page. | "Preprints, talks, and vulnerability disclosures are allowed." | https://www.usenix.org/conference/sais27/call-for-papers |
| MLSys 2027, Bellevue WA, Jun 21-25 2027 (main conference) | Paper 2026-10-30 (CFP page: 20:00 UTC; Dates page: 12:00 PM PDT; the two official pages disagree by one hour). No separate abstract deadline. Submission via OpenReview; every author needs an active OpenReview profile. Author response 2027-01-21; notification 2027-02-28. | 10 pages excluding references; appendices unlimited (separate upload); MLSys style files, double-blind. | Voluntary; follows ACM badging; does not affect the accept decision. | "Authors are allowed to post their papers on arXiv or other public forums." | https://mlsys.org/Conferences/2027/CallForResearchPapers and https://mlsys.org/Conferences/2027/Dates |

Fit notes (facts, not a recommendation):

- USENIX Security and SAIS read the paper as a security-systems contribution
  (verifiable decode receipt, independent verifier, threat model). Both require
  the Open Science Appendix, which the repository's public logs, golden files
  and verifier already serve.
- MLSys reads it as a systems contribution (integer-only inference, cross-ISA
  identity, bare-metal boot). Its deadline is three days after this file's date
  and its limit is 10 pages, so it is reachable only with a cut-down draft
  prepared in that window.
- NDSS 2027 (main) is closed for this cycle: both listed deadlines (2026-05-06
  and 2026-08-19 AoE) have passed and no later cycle is posted
  (https://www.ndss-symposium.org/ndss2027/submissions/call-for-papers/).

## Watch list: fitting venues with no 2027 CFP posted yet

| Venue | Previous cycle | What is known | Source |
|---|---|---|---|
| EuroMLSys 2027 (EuroSys workshop; EuroSys 2027 Rabat, Apr 19-23 2027) | 2026 edition: paper 2026-02-15 AoE (second date 2026-02-24). | 6 pages excluding bibliography; SIGPLAN template; single-blind. AE and preprint policy not stated. | https://euromlsys.eu/ (2026 page) |
| EdgeSys 2027 (EuroSys workshop) | 2026 edition: paper 2026-03-22 AoE. | Format, AE and preprint rules on a separate submission page, not read. | https://edge-sys.github.io/2026/ |
| ACM CCS 2027 workshops (Atlanta, Oct 11-15 2027) | CCS 2026 workshops (closed). | CCS 2027 page says "Coming Soon"; no workshop calls. | https://sigsac.org/ccs/CCS2027/ |
| NDSS 2027 co-located workshops | Workshop-proposal call only (proposals were due 2026-09-01). | No accepted workshops and no workshop paper CFPs listed as of the fetch date. | https://www.ndss-symposium.org/ndss2027/submissions/call-for-co-located-workshops/ |
| USENIX WOOT '27 (Denver, Aug 9-10 2027) | WOOT '26 had two cycles (reported, not read on the venue page). | No CFP content on the WOOT '27 page. | https://www.usenix.org/conference/woot27 |
| tinyML / EDGE AI Research Track 2027 | 2025 edition (reported): 6 pages, double-blind, proceedings on arXiv. | No 2027 page found. | search results only; no venue page fetched |

## arXiv preprint: gating item and fallbacks

- Gating item: arXiv requires endorsement before a user's initial submission to
  arXiv or to a new category; "at least one positive endorsement is required per
  endorsement category" (https://info.arxiv.org/help/endorsement.html, undated).
  The automatic route needs an institutional e-mail plus a claimed co-authored
  paper; the personal route needs an eligible endorser (papers in the same
  endorsement domain submitted between three months and five years ago).
- Status: a submission draft exists and is blocked on endorsement. The page does
  not say whether an endorsement for one category (cs.LG) covers another
  (cs.CR), and does not name either category; a cs.CR-primary listing may need
  its own endorsement.
- Fallbacks if endorsement does not arrive:
  1. Zenodo DOI deposit: no deadline, "any status is accepted, from any stage of
     the research lifecycle", no affiliation requirement stated
     (https://about.zenodo.org/policies/, "General Policies v1.0", undated).
     Whether a Zenodo record counts as prior publication at a given venue was
     not checked.
  2. OpenReview: independent researchers can register (personal-e-mail profiles
     are moderated; an institutional e-mail speeds activation). Posting a public
     preprint outside any venue was not found in the OpenReview documentation
     read (https://docs.openreview.net/getting-started/frequently-asked-questions/i-am-an-independent-researcher-how-do-i-sign-up.md).
  3. Direct submission: all three candidate venues above accept submissions
     without a preprint, and all three allow a preprint during review.

## Not checked on 2026-10-27

- Page 2 of the USENIX upcoming-events list (a trustworthy- or
  verifiable-computing workshop there would be unchecked).
- The USENIX Security 2027 call-for-artifacts page (returned 404).
- EdgeSys 2026 submission page (format, AE, preprint rules).
- Prior-publication rules for Zenodo records at each venue.
