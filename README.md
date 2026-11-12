# Helminth / *Bordetella pertussis* interaction model

Code for the model presented in Appendix A3: an age- and worm-burden-structured
model of pertussis transmission in a helminth-endemic setting. All outputs are
computed from the parameters defined in the scripts; no external data are needed.

## Files

| File | Content |
|---|---|
| `hp_model.py` | Model definition (structure, calibration, equilibrium solver, outcomes). See its docstring for the full description. |
| `results.py` | Numbers quoted in the manuscript and in Appendix A3. |
| `figure_A1.py` | Figure A1 (helminth control, δ, infant infections, deworming benefit) and the values plotted. |
| `sensitivity.py` | Global sensitivity analysis (Latin hypercube sampling, 12 parameters). |
| `figure_A2.py` | Figure A2 and summary of the sensitivity analysis (PRCC, distribution of the effect). |

## Requirements

Python ≥ 3.10 with numpy, scipy, pandas and matplotlib:

```
pip install -r requirements.txt
```

Tested with Python 3.14.6, numpy 2.4.4, scipy 1.17.1, pandas 3.0.3, matplotlib 3.10.9.

## Reproducing the results

Run from this folder (`OUT` is the output folder):

```
python results.py OUT/results.json                              # ~10 s
python figure_A1.py OUT                                         # ~20 s
python sensitivity.py 500 OUT/sensitivity_lhs.csv               # ~3 min (2 processes)
python figure_A2.py OUT/sensitivity_lhs.csv OUT                 # a few seconds
```

Outputs:

- `results.json`
- `FigureA1_model.png`, `FigureA1_model.pdf`, `FigureA1_values.json`
- `sensitivity_lhs.csv` (one row per parameter set)
- `FigureA2_sensitivity.png`, `FigureA2_sensitivity.pdf`, `sensitivity_summary.json`

The Latin hypercube uses a fixed seed (`seed=1`), so all results are
deterministic and reproducible up to floating-point rounding. Minor cosmetic
differences in the figures (e.g. tick spacing) may appear with other
matplotlib versions.
