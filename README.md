# PRISM-Gen

**Physics-guided Robust Inhibitor Selection Method — Generative Module**

A multi-fidelity computational framework for the discovery of broad-spectrum coronavirus main protease (Mpro) inhibitors.

---

## Overview

**PRISM-Gen** integrates generative molecular design with hierarchical quantum-chemical screening and multi-target docking to prioritize candidate inhibitors with predicted cross-strain binding competence against coronavirus M<sup>pro</sup> orthologs.

### Pipeline Architecture

The framework follows a seven-stage multi-fidelity screening strategy:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | FRATTVAE | Fragment-tree variational autoencoder for molecular generation |
| 2 | Uni-Mol surrogate model | Predicted inhibitory activity scoring |
| 3 | Latent-space optimization | Reward-guided exploration of chemical space |
| 4 | GFN2-xTB + GEM | Semi-empirical electronic screening with Gaussian Electronic Moderation |
| 5 | ADMET filtering | Drug-likeness and pharmacokinetic assessment |
| 6 | B3LYP/6-31G* DFT | High-level electronic-structure validation |
| 7 | Multi-target docking | Worst-case broad-spectrum scoring across SARS-CoV-2, SARS-CoV-1, and MERS-CoV M<sup>pro</sup> |

### Key Features

- **Multi-fidelity electronic screening:** Three-tier cascade (xTB → GEM → DFT) balancing computational cost and physical accuracy
- **Gaussian Electronic Moderation (GEM):** Continuous scoring that re-ranks candidates rather than applying hard electronic cutoffs, preserving scaffold diversity
- **Worst-case broad-spectrum docking:** Candidates ranked by weakest cross-target affinity (Score<sub>broad</sub> = max E<sub>i</sub>), selecting for consistent multi-target binding
- **Generator-agnostic architecture:** Downstream screening stages are decoupled from the generator and accept SMILES from any source

---

## Repository Structure

```
PRISM-Gen/
├── pipeline/                    # Core pipeline source code
│   ├── core/                    # Step-by-step implementation (Steps 1–6)
│   │   ├── step1_vae.py         # FRATTVAE molecular generation
│   │   ├── step2_surrogate.py   # Surrogate activity prediction
│   │   ├── step3a_optimize.py   # Latent-space optimization
│   │   ├── step3b_run_dft.py    # GFN2-xTB electronic screening
│   │   ├── step3c_dft_refine.py # GEM scoring and re-ranking
│   │   ├── step4a_admet.py      # ADMET filtering
│   │   ├── step4b_final_pyscf.py# B3LYP/6-31G* DFT validation
│   │   ├── step5a_docking.py    # Multi-target broad-spectrum docking
│   │   └── ...
│   ├── data/                    # Input data and receptor structures
│   ├── models/                  # Pretrained model checkpoints
│   ├── results/                 # Pipeline output directory
│   ├── environment.yml          # Conda environment specification
│   ├── requirements.txt         # Pip dependency list
│   └── INSTALL.md               # Detailed installation guide
├── openclaw-skill/              # ClawHub interactive demo skill
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/SenaZeng/PRISM-Gen.git
cd PRISM-Gen

# Create conda environment
conda env create -f pipeline/environment.yml
conda activate ai_drug_design
```

For detailed installation instructions, dependency versions, and troubleshooting, see [`pipeline/INSTALL.md`](pipeline/INSTALL.md).

### 2. Download Pretrained Checkpoint

The FRATTVAE checkpoint (~125 MB) is archived on Zenodo:

```bash
wget https://zenodo.org/records/18764996/files/model_best.pth -O pipeline/pretrained_model/model/model_best.pth
```

**Zenodo DOI:** [10.5281/zenodo.18764996](https://doi.org/10.5281/zenodo.18764996)

### 3. Run the Pipeline

```bash
cd pipeline

# Run the full pipeline (Steps 1–6)
python core/run_pipeline.py

# Or run individual stages
python core/step1_vae.py                  # Molecular generation
python core/step3b_run_dft.py             # xTB electronic screening
python core/step4b_final_pyscf.py         # DFT validation
python core/step5a_docking.py \
    --receptor_dir data/receptors \
    --top_n 36                            # Multi-target docking
```

---

## Data Sources

| Data | Source | Usage |
|------|--------|-------|
| M<sup>pro</sup> activity labels | [TDC SARS-CoV-2 M<sup>pro</sup> benchmark](https://tdcommons.ai/) | Surrogate model training |
| hERG safety labels | [TDC hERG central dataset](https://tdcommons.ai/) | Cardiac toxicity filtering |
| SARS-CoV-2 M<sup>pro</sup> | [PDB: 6W63](https://www.rcsb.org/structure/6W63) | Docking target |
| SARS-CoV-1 M<sup>pro</sup> | [PDB: 3V3M](https://www.rcsb.org/structure/3V3M) | Docking target |
| MERS-CoV M<sup>pro</sup> | [PDB: 4YLU](https://www.rcsb.org/structure/4YLU) | Docking target |

---

## Interactive Demo

An interactive analysis tool for exploring pre-calculated screening results is available on ClawHub:

🔗 [https://clawhub.ai/SenaZeng/prism-gen-demo](https://clawhub.ai/SenaZeng/prism-gen-demo) (v1.0.4)

Source files for the demo skill are provided in the [`openclaw-skill/`](openclaw-skill/) directory.

---

## Key Software Versions

| Software | Version |
|----------|---------|
| Python | 3.9 |
| RDKit | 2025.03.2 |
| PySCF | 2.11.0 |
| GFN2-xTB | 6.7.1 |
| AutoDock Vina | 1.1.2 |
| PyTorch | 1.12.1 (CUDA 11.3) |
| DGL | 0.9.1.post1 |

See [`pipeline/INSTALL.md`](pipeline/INSTALL.md) for the complete dependency list.

---

## Citation

If you use PRISM-Gen in your research, please cite:

> [Citation will be added upon publication]

---

## License

This repository is released for **academic research purposes only**.
Any commercial use of this software requires explicit permission from the author.

---

## Contact

For questions or issues, please open a [GitHub Issue](https://github.com/SenaZeng/PRISM-Gen/issues).
