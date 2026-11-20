# c(k) Wing Trade Law and Penguin BE Wingtip Retrofit

[![DOI](https://zenodo.org/badge/1465262967.svg)](https://doi.org/10.5281/zenodo.24393993)

A longer wing makes less induced drag but bends its structure harder. This project asks how much extra span an *existing* wing can afford within its current bending limits, and answers it in two parts:

1. **Theory.** From Prandtl's lifting-line theory, the least induced-drag penalty for holding a bending measure with moment arm |y|^k to a fraction *f* of its elliptical value is

   **C_Di / C_Di,ell = 1 + c(k)(1 − f)²,  with c(k) = 4(1 + k)/k²**

   verified by a 151-harmonic optimiser and by direct summation.
2. **Application.** A bolt-on wingtip retrofit for the Penguin BE survey UAV, using the existing ailerons for load alleviation. 4,890 designs were screened with a lifting-line model, re-checked with a vortex-lattice model, and the finalists checked with viscous drag.

## Key result

**Recommended wing W2:** +206 mm per tip (span 3.30 → 3.71 m), −4° washout, alleviation gain 7.5°/g, +0.217 kg.

| | Result |
|---|---|
| Induced drag | −21.1% (lifting line), −20.6% (vortex lattice) |
| Cruise endurance | +4.3% (viscous) to +5.9% (lifting line) |
| Root / joiner bending at 3.8 g | 0.978 / 0.977 of the original rating |

The vortex lattice rejected 295 designs that lifting line had passed, including the lifting-line winner: lifting line overstates the bending relief from deflected ailerons.

![W2 planform](figures/final_wing_W2_planform.png)

## Repository contents

| Path | What it is |
|---|---|
| `ck-and-penguin-be-wing-study.ipynb` | The full notebook: Part A (theory), Part B (retrofit) |
| `wingstudy/` | Model code (`wing.py`, `study.py`, `rank.py`, `final.py`, …) and Stage 1 results |
| `figures/`, `results/` | Stage 2 and 3 outputs and final figures |
| `requirements.txt` | Package versions used |

## How to run

```bash
pip install -r requirements.txt
```

Open the notebook and run all cells in order (on Kaggle, turn Internet on). Stage 1 takes about a minute per scenario. Stage 2 takes about 1 hour on a CPU; it resumes from `wingstudy/results/vlm_screen_raw.csv` if that file exists, so a fresh run from this repo skips it.

## Main assumptions

Straight-tapered planform approximating the real wing; NACA 2412 sections; rigid wing; aileron modelled as twist (τ = 0.5); instantaneous alleviation; whole-aircraft L/D = 18. No physical tests yet.

## Paper and related work

- Paper: [engrXiv preprint, DOI pending]
- Earlier theory-only repository: [wing-loading-tradeoff](https://github.com/Manthonix/wing-loading-tradeoff)

## Citation and licence

Cite via `CITATION.cff` or the Zenodo DOI above. Code: MIT. Data and figures: CC BY 4.0.

**Author:** Manthan Rana, Independent Researcher, Mumbai, India
