# Literature to add — synthetic EV charging data generators

Papers surfaced during baseline/lit-review discussion that are **NOT yet cited**
in the current draft. Grouped by role, with modality and relevance noted.

Legend for *Output*: **session** = tabular session attributes (arrival, duration,
energy) — directly comparable to our method; **load-TS** = charging power / load
time-series; **sim** = bottom-up simulator (not data-fitted); **forecast** =
load forecasting (not generation).

---

## A. Candidate baselines (data-fitted, session-level → directly comparable)

| # | Method / paper | Year | Approach | Output | Link |
|---|----------------|------|----------|--------|------|
| 1 | **CTGAN / TVAE** — "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS) | 2019 | conditional tabular GAN + tabular VAE | session | https://arxiv.org/abs/1907.00503 |
| 2 | **TabDDPM** — "Modelling Tabular Data with Diffusion Models" | 2022 | denoising diffusion for tabular data (modality-matched diffusion baseline) | session | https://arxiv.org/abs/2209.15421 |
| 3 | **Vine copula** — "Multivariate copula procedure for EV charging event simulation" (Energy) | 2022 | pair-copula construction over arrival/duration/energy; Student-t best | session | https://www.sciencedirect.com/science/article/pii/S0360544221019666 |
| 4 | **Copulas → Neural Density Estimation for EV charging events** | 2026 | vine copulas vs neural density estimation on EV events | session | https://arxiv.org/pdf/2603.29554 |
| 5 | **Joint EV driving & charging via deep generative networks** (Transp. Research Part C) | 2025 | Transformer + GMM + deep Gibbs sampling; spatiotemporal | session (+driving) | https://www.sciencedirect.com/science/article/abs/pii/S0968090X25004851 |
| 6 | **CTGAN + KDE generator** (preprint) — Caltech/ACN | 2025 | CTGAN for inter-column structure + KDE for connection time | session | https://www.preprints.org/manuscript/202508.1262/v1/download |
| 7 | **ev-flow** — NHTS-grounded generator, 8 U.S. regions | 2026 | reproducible, survey-grounded behavior generator | session/behavior | https://arxiv.org/pdf/2606.19520 |
| 8 | **VAE for EV load profiles** (Energy 12(5):849) | 2019 | variational autoencoder (EV-domain VAE precedent) | load-TS/session | https://www.mdpi.com/1996-1073/12/5/849/htm |
| 8a | **Synthesis of EV charging data: A real-world data-driven approach** — Li, Bian, Chen, Ozbay, Zhong (*Communications in Transportation Research* 4(3):100128, open access) | 2024 | conditional density networks (mixture-density → GMM params) + Gibbs sampling; 1.65M events / 3,777 BEVs Shanghai; conditional/future generation | session/event | https://doi.org/10.1016/j.commtr.2024.100128 |

## B. Related work — generative EV (different modality; cite, don't necessarily benchmark)

| # | Method / paper | Year | Approach | Output | Link |
|---|----------------|------|----------|--------|------|
| 9  | **DiffCharge** — Generating EV Charging Scenarios via Denoising Diffusion (IEEE TSG) | 2023/24 | denoising diffusion; self-attention; type-conditioned | load-TS | https://arxiv.org/abs/2308.09857 |
| 10 | **DiffPLF** — Conditional Diffusion for Probabilistic Forecasting of EV Charging Load | 2024 | conditional diffusion (forecasting) | forecast | https://arxiv.org/abs/2402.13548 |
| 11 | **EnergyDiff** — Universal Time-Series Energy Data Generation using Diffusion | 2024 | diffusion for energy time-series (incl. EV/residential) | load-TS | https://arxiv.org/pdf/2407.13538 |
| 12 | **AI-Augmented Multi-Prototype EV Charging Load Profiles (China)** (Nature Sci. Data) | 2026 | generative-AI-augmented load-profile dataset | load-TS | https://www.nature.com/articles/s41597-026-07273-5 |
| 13 | **LLM-Enabled Frequency-Aware Flow-Diffusion** — NL-guided power-system scenarios | 2026 | LLM + flow-diffusion (EV-adjacent) | time-series | https://arxiv.org/pdf/2602.19522 |

## C. Simulators & datasets (bottom-up / assumption-driven; cite as different class)

| # | Method / paper | Year | Approach | Output | Link |
|---|----------------|------|----------|--------|------|
| 14 | **FlexiGen** — Stochastic Dataset Generator for EV Charging Energy Flexibility | 2024 | configurable stochastic simulator (V1G/V2G, SoC routines) | sim | https://arxiv.org/abs/2411.07040 |

## D. Surveys (for related-work framing)

| # | Paper | Year | Link |
|---|-------|------|------|
| 15 | **Electric Vehicle Charging Load Modeling: A Survey, Trends, Challenges and Opportunities** | 2025 | https://arxiv.org/abs/2511.03741 |
| 16 | **EV Behavior Modeling for Vehicle-to-Grid Integration: Methods, Challenges, and Perspectives** — Zhao et al., *Energies* 19(4):871 (review; statistical/data-driven/decision-oriented paradigms; Task–Data–Deployment framework; flags data heterogeneity & cross-scenario generalizability) | 2026 | https://doi.org/10.3390/en19040871 |

---

## Verified details (fetched this session)

- **EV-SDG (2020)** — Lahariya, D. F. Benoit, C. Develder (Ghent/imec-IDLab).
  Conference: ACM e-Energy 2020; journal: *Energies* 13(16):4211. Exponential
  inter-arrival + conditional GMM (connection time & energy). *Already our baseline.*
- **Joint driving+charging (2025)** — deep generative: Transformer + GMM + deep Gibbs
  sampling; joint sequential driving+charging events; per-user preference-optimization
  fine-tuning. **Venue: Transportation Research Part C (Elsevier), Dec 2025.** Comparable
  (heavier) deep-generative baseline.
- **CTGAN + KDE (2025)** — CTGAN (inter-column structure) + KDE (connection time),
  Caltech/ACN, 185 days → ~29,600 days. **Preprint (preprints.org / MDPI), Aug 2025**
  (not peer-reviewed).
- **Copulas → Neural (2026)** — Výboh & Grmanová (KInIT, Slovakia). Vine copulas + CODINE
  vs 6 parametric copulas + GMMNet, on arrival/duration/energy. Datasets: Slovakia
  residential, **Trondheim/Sørensen Norway (SAME as ours)**, Dundee public. IEEE-style,
  arXiv 2603.29554. *Highly relevant — essentially the vine/neural baseline study.*
- **ev-flow (2026)** — Travacca. **RECLASSIFIED: bottom-up, NHTS-survey-grounded US
  generator (emobpy/VencoPy family), NOT fit-to-data.** arXiv 2606.19520. → related work,
  not a drop-in baseline.

## Notes for the revision

- **Baseline suite (comparable, data-fitted, session-level):** EV-SDG (already cited)
  + Gaussian copula/GMM + Vine copula (#3/#4) + CTGAN/TVAE (#1) — optionally TabDDPM (#2)
  as the modality-matched diffusion baseline.
- **Position DiffCharge / DiffPLF / EnergyDiff / AI-augmented profiles (#9–12)** as
  *load–time-series* generators — a different modality from our session-attribute setting;
  one sentence in related work preempts the reviewer question.
- **Position FlexiGen (#14)** as a bottom-up *simulator* (not fit to data), alongside EV-SDG's
  parametric family.
- **Survey (#15)** is a good anchor citation for the "generative models for EV data" paragraph.
- Author lists are intentionally omitted where not verified — confirm from each source before
  inserting BibTeX.
