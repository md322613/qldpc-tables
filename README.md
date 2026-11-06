# qldpc-tables

Certified parameters for best-known quantum LDPC codes at hardware-relevant sizes.

Every code is rebuilt from its published polynomials, n and k are recomputed by GF(2) rank, and the minimum distance is computed with a mixed-integer program (HiGHS) for both X- and Z-type logicals. Translation symmetry of the abelian group acts transitively on each block, so a minimum-weight logical may be assumed to contain the first qubit of the first block it touches; this splits each search into two smaller exact programs.

Each result in `results/` records:
- `d.upper`: weight of a stored logical operator (the witness, re-checked against the parity checks);
- `d.lower`: the solver's certified lower bound; `exact` when the two agree;
- per-case solver status, dual bounds and timings, and the URL of the CI run that produced it.

Runs happen in public GitHub Actions (`.github/workflows/verify.yml`), one job per code, so every certificate has an auditable log. Add a code to `codes.json` and push, or start the workflow manually with a subset of names.

Code families currently covered: bivariate bicycle codes (Bravyi et al., arXiv:2308.07915), weight-8 BB codes (Lu–Yang–Guo, arXiv:2609.06572), cover codes (Symons–Rajput–Browne, arXiv:2511.13560), self-dual BB codes on twisted tori (Liang–Chen, arXiv:2510.05211).

Local run: `pip install numpy galois highspy && python qverify.py "LYG T1#6" 3600 4`
