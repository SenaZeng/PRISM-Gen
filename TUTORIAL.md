# PRISM-Gen: Adapting the Pipeline to a New Protein Target

This note describes the minimal changes required to apply PRISM-Gen
to a protein target other than coronavirus Mpro.

For installation and reproduction of paper results, see `README.md`.

> **Note on configuration:** `run_pipeline.py` does not use command-line
> arguments. All pipeline parameters are hardcoded constants at the top
> of that file. Edit them directly before running.

---

## What Needs to Change

| Component | File | What to replace or adjust |
|---|---|---|
| Receptor structure | `data/receptors/` | Replace with target PDB/PDBQT |
| Docking box | `step5a_docking.py` | Update `TARGET_CONFIG` center + box size |
| Activity surrogate | `step2_surrogate.py` | Edit hardcoded data path; retrain |
| MW window | `step3a_optimizer.py` | Change `mw_min` / `mw_max` defaults in source |
| hERG threshold | `run_pipeline.py` | Edit `HERG_THRESHOLD_ADMET` constant |

Components that **do not** require changes:
FRATTVAE generator, GEM scoring, xTB descriptors, DFT level of theory.

---

## Step 1 — Prepare the Receptor

Download the target structure from [RCSB PDB](https://www.rcsb.org/).
Convert to PDBQT using AutoDock Tools or Open Babel:

```bash
obabel YOUR_TARGET.pdb -O YOUR_TARGET.pdbqt -xr --partialcharge gasteiger
```

Update `TARGET_CONFIG` in `step5a_docking.py` with the binding-site
center coordinates (obtainable from PyMOL using the co-crystallized ligand centroid)
and the path to the new PDBQT file:

```python
# In step5a_docking.py
TARGET_CONFIG = {
    "YOUR_TARGET": {
        "pdbqt":    "data/receptors/YOUR_TARGET.pdbqt",
        "center":   (X, Y, Z),          # Å, binding-site centroid
        "box_size": (24.0, 24.0, 24.0), # Å, adjust for larger pockets
    }
}
```

---

## Step 2 — Prepare Activity Training Data

Obtain IC50 or Ki data from ChEMBL, BindingDB, or TDC.
Format as a two-column CSV (`smiles`, `pIC50`).

Then edit `step2_surrogate.py` directly to point to your new data file
and retrain:

```python
# In step2_surrogate.py, update the hardcoded data path:
data_file = "data/YOUR_TARGET_activity.csv"

# Also update the model save path inside SurrogateModel.__init__:
self.sklearn_model_path = "results/surrogate_YOUR_TARGET.pkl"
```

Run the script standalone to train:

```bash
python core/step2_surrogate.py
```

---

## Step 3 — Adjust Key Parameters

**Pipeline-level constants** (edit at the top of `run_pipeline.py`):

```python
# run_pipeline.py — edit these constants directly
RESTARTS  = 100   # number of hill-climbing restarts
STEPS     = 100   # steps per restart
HERG_THRESHOLD_ADMET = 0.5   # hERG hard-filter cutoff (Step 4A)
```

**MW window** (edit defaults in `step3a_optimizer.py`):

```python
# In step3a_optimizer.py — update argparse defaults
parser.add_argument("--mw_min", type=float, default=320.0, ...)
parser.add_argument("--mw_max", type=float, default=520.0, ...)
```

Set `mw_min` / `mw_max` to bracket the MW distribution of known actives
for your target (typically 250–550 Da; CNS targets: 250–400 Da).
All other reward weights (QED, SA, LogP) can remain at their defaults
for an initial run.

---

## Step 4 — Run the Pipeline

```bash
cd /path/to/project_root
python core/run_pipeline.py
```

No command-line flags are needed; all parameters are read from the
constants defined at the top of `run_pipeline.py`.

The primary output is `results/step5b_final_candidates.csv`.
For a single-target application, rank by the target-specific `E_*`
docking score column directly rather than `Broad_Spectrum_Score`.

---

For questions, open a [GitHub Issue](https://github.com/SenaZeng/PRISM-Gen/issues).
