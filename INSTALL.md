# PRISM-Gen — Installation Guide

## System Requirements

| Item | Specification |
|------|--------------|
| Operating System | Linux (tested on CentOS 7 / Ubuntu 20.04+) |
| Python | 3.9.x |
| CUDA | 11.3 (for GPU-accelerated surrogate model training) |
| RAM | ≥ 32 GB recommended |
| Disk | ≥ 10 GB (including model checkpoints and intermediate results) |

---

## Key Dependencies and Versions

The table below lists the core scientific packages used in the PRISM-Gen pipeline.
Full dependency lists are provided in `environment.yml` (conda) and `requirements.txt` (pip).

| Package | Version | Role in Pipeline |
|---------|---------|-----------------|
| Python | 3.9.23 | Runtime |
| RDKit | 2025.03.2 (conda) | Molecular representation, fingerprints, Bemis–Murcko scaffolds |
| PySCF | 2.11.0 | B3LYP/6-31G* DFT validation (Stage 5) |
| xtb (GFN2-xTB) | 6.7.1 | Semi-empirical electronic screening (Stage 3) |
| AutoDock Vina | 1.1.2 | Multi-target molecular docking (Stage 7) |
| Meeko | 0.7.1 | Ligand preparation for Vina |
| Open Babel | 3.1.1 | Molecular format conversion |
| PyTorch | 1.12.1+cu113 | FRATTVAE generator and surrogate model |
| DGL (Deep Graph Library) | 0.9.1.post1 (cu113) | Graph neural network operations |
| Uni-Mol Tools | 0.1.4.post1 | Molecular representation for surrogate scoring |
| NumPy | 1.26.4 | Numerical computation |
| pandas | 1.3.5 | Data manipulation |
| scikit-learn | 1.6.1 | Machine learning utilities |
| Matplotlib | 3.9.4 | Visualization |
| Seaborn | 0.13.2 | Statistical visualization |
| BioPython | 1.85 | Protein structure parsing |
| PyTDC | 0.3.8 | Therapeutics Data Commons benchmark access |
| tqdm | 4.67.1 | Progress bars |
| joblib | 1.5.1 | Parallel computation |
| OpenPyXL | 3.1.5 | Excel I/O for result tables |

---

## Installation

### Option A: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/SenaZeng/PRISM-Gen.git
cd PRISM-Gen

# Create the conda environment from the specification file
conda env create -f pipeline/environment.yml

# Activate the environment
conda activate ai_drug_design
```

### Option B: Pip Installation

```bash
# Create and activate a virtual environment
python3.9 -m venv prism_env
source prism_env/bin/activate

# Install dependencies
pip install -r pipeline/requirements.txt
```

> **Note:** Some packages (e.g., `xtb`, `autodock-vina`, `openbabel`, `rdkit`) are
> more reliably installed via conda. If using pip, you may need to install these
> separately through conda-forge:
> ```bash
> conda install -c conda-forge xtb=6.7.1 autodock-vina=1.1.2 openbabel=3.1.1 rdkit=2025.03.2
> ```

### PyTorch with CUDA 11.3

If PyTorch is not installed by the environment file, install manually:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

pip install dgl-cu113==0.9.1.post1 -f https://data.dgl.ai/wheels/repo.html
```

---

## Pretrained Model Checkpoint

The FRATTVAE pretrained checkpoint (`model_best.pth`, ~125 MB) is archived on Zenodo:

**DOI:** [10.5281/zenodo.18764996](https://doi.org/10.5281/zenodo.18764996)

Download and place it in the appropriate directory:

```bash
# Download from Zenodo
wget https://zenodo.org/records/18764996/files/model_best.pth

# Place in the model directory
mv model_best.pth pipeline/results/pretrained_model/model/
```

---

## Receptor Structures

The three Mpro crystal structures used for multi-target docking are retrieved from the
Protein Data Bank:

| Target | PDB ID | Organism |
|--------|--------|----------|
| SARS-CoV-2 Mpro | [6W63](https://www.rcsb.org/structure/6W63) | SARS-CoV-2 |
| SARS-CoV-1 Mpro | [3V3M](https://www.rcsb.org/structure/3V3M) | SARS-CoV-1 |
| MERS-CoV Mpro | [4YLU](https://www.rcsb.org/structure/4YLU) | MERS-CoV |

Prepared receptor files (`.pdbqt`) are provided in `pipeline/data/receptors/`.
Preparation details are described in the accompanying manuscript.

---

## Verification

After installation, verify the environment:

```bash
conda activate ai_drug_design

python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import rdkit; print(f'RDKit: {rdkit.__version__}')
import pyscf; print(f'PySCF: {pyscf.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
"

# Verify xtb
xtb --version

# Verify Vina
vina --version
```

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `GLIBCXX_X.X.XX not found` | Ensure the conda environment's `lib/` is in `LD_LIBRARY_PATH` |
| xtb or Vina not found | Verify conda-forge installation: `conda list xtb` |
| CUDA out of memory | Reduce `--workers` or `--batch_size` in generation/optimization steps |
| PySCF slow on DFT | Set `export OMP_NUM_THREADS=1` and increase `--workers` for parallel multi-process execution |

---

## License

This repository is released for **academic research purposes only**.
Any commercial use requires explicit permission from the author.
