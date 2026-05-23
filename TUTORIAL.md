# PRISM-Gen: Adapting the Pipeline to a New Protein Target

This note describes the minimal changes required to apply PRISM-Gen
to a protein target other than coronavirus Mpro.

For installation and reproduction of paper results, see `README.md`.

---

## What Needs to Change

| Component | File | What to replace or adjust |
|---|---|---|
| Receptor structure | `data/receptors/` | Replace with target PDB/PDBQT |
| Docking box | `step5a_docking.py` | Update `TARGET_CONFIG` center + box size |
| Activity surrogate | `step2_surrogate.py` | Retrain on target-specific IC50/Ki data |
| MW window | `step3a_optimizer.py` | Set `--mw_min` / `--mw_max` per target |
| hERG threshold | `run_pipeline.py` | Adjust `HERG_THRESHOLD_ADMET` if needed |

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
and the path to the new PDBQT file.

---

## Step 2 — Prepare Activity Training Data

Obtain IC50 or Ki data from ChEMBL, BindingDB, or TDC.
Format as a two-column CSV (`smiles`, `pIC50`) and retrain the surrogate:

```bash
python step2_surrogate.py \
    --input  data/YOUR_TARGET_activity.csv \
    --output models/surrogate_YOUR_TARGET.pkl \
    --target pIC50
```

Update `SURROGATE_MODEL_PATH` in `run_pipeline.py` accordingly.

---

## Step 3 — Adjust Key Parameters

For most targets, only the molecular weight window requires adjustment.
Set `--mw_min` and `--mw_max` to bracket the MW distribution of known
actives for the target (typically 250–550 Da; CNS targets: 250–400 Da).

All other reward weights (QED, SA, LogP) can remain at their defaults
for an initial run and be refined based on the output distribution.

---

## Step 4 — Run the Pipeline

Quick validation (before full run):

```bash
python run_pipeline.py \
    --n_restarts 5 --steps 20 --top_k 20 \
    --skip_dft True --demo_mode True
```

Full run:

```bash
python run_pipeline.py \
    --n_restarts 100 --steps 100 --top_k 200 \
    --dft_top_k 46 --final_top_k 36 --n_jobs 40
```

The primary output is `results/step5b_final_candidates.csv`.
For a single-target application, rank by `E_YOUR_TARGET` directly
rather than `Broad_Spectrum_Score`.

---

For questions, open a [GitHub Issue](https://github.com/SenaZeng/PRISM-Gen/issues).
