# File: core/step3b_run_dft.py
# -*- coding: utf-8 -*-
"""
Step 3B: Parallel xTB electronic structure pre-screening

- argparse handles command-line arguments (input/output files, top_k, worker count)
- concurrent.futures.ProcessPoolExecutor provides multi-process parallelism
- Core logic (parse_xtb_output, get_best_conformer, etc.) is preserved from the original
- Progress reporting prevents cluttered console output during parallel execution
"""


import os
import sys
import subprocess
import re
import tempfile
import argparse
import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

# ================= Default configuration (overridable via command-line arguments) =================
DEFAULT_INPUT_FILE = "../results/step3a_optimized_molecules.csv"
DEFAULT_OUTPUT_FILE = "../results/step3b_dft_results.csv"
DEFAULT_TOP_K = 200
DEFAULT_WORKERS = 40
XTB_EXE = os.environ.get("XTB_EXE", "xtb")
XTB_SOLVENT = "water"
NUM_CONFS = 20
# ==================================================================================================

def parse_xtb_output(text: str):
    """
    Parse HOMO / LUMO / gap (eV) from xTB standard output text.
    """
    homo_ev = None
    lumo_ev = None
    gap_ev = None

    lines = text.splitlines()

    for line in lines:
        # Format A: legacy format
        if "HOMO orbital eigv." in line:
            m = re.search(r"([-+]?\d+\.\d+)\s*eV", line)
            if m: homo_ev = float(m.group(1))
        if "LUMO orbital eigv." in line:
            m = re.search(r"([-+]?\d+\.\d+)\s*eV", line)
            if m: lumo_ev = float(m.group(1))
        if "HL-Gap" in line or "HOMO-LUMO gap" in line:
            m = re.search(r"([-+]?\d+\.\d+)\s*eV", line)
            if m: gap_ev = float(m.group(1))

        # Format B: newer format with (HOMO) / (LUMO) tags
        if "(HOMO)" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums: homo_ev = float(nums[-1])
        if "(LUMO)" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums: lumo_ev = float(nums[-1])

    if gap_ev is None and (homo_ev is not None and lumo_ev is not None):
        gap_ev = lumo_ev - homo_ev

    return homo_ev, lumo_ev, gap_ev


def parse_xtb_min_charge(text: str):
    """Parse the most negative atomic charge from xTB output."""
    min_charge = None
    lines = text.splitlines()
    in_block = False

    for line in lines:
        if not in_block:
            if "covCN" in line and "q" in line:
                in_block = True
                continue
        else:
            stripped = line.strip()
            if not stripped or stripped.startswith("---") or "sum of atomic charges" in stripped.lower() or stripped.startswith("#"):
                in_block = False
                continue
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if len(nums) >= 2:
                try:
                    charge = float(nums[1])
                except ValueError:
                    continue
                if min_charge is None or charge < min_charge:
                    min_charge = charge
    return min_charge


def get_best_conformer(mol: Chem.Mol, num_confs: int = NUM_CONFS) -> Chem.Mol:
    """Generate multiple conformers and select the lowest-energy one."""
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDG()
        params.randomSeed = 42
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if not cids:
            raise RuntimeError("EmbedMultipleConfs failed")

        res = []
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None: continue
            ff.Minimize()
            res.append((ff.CalcEnergy(), cid))

        if not res: raise RuntimeError("MMFF calculation failed")
        
        res.sort(key=lambda x: x[0])
        best_cid = res[0][1]
        
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(mol.GetConformer(best_cid), assignId=True)
        return new_mol

    except Exception:
        # Fallback: single conformer
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        return mol


def run_xtb_task(args):
    """
    Single-molecule xTB calculation task (intended for parallel invocation).
    args: (smiles, idx, mock_mode)
    """
    smiles, idx, mock_mode = args
    
    # Mock mode (for pipeline testing only)
    if mock_mode:
        gap = np.random.normal(5.0, 1.5)
        homo = -6.0 - (gap / 2.0)
        return {
            "smiles": smiles,
            "EHOMO_ev": float(homo),
            "ELUMO_ev": float(homo + gap),
            "gap_ev": float(gap),
            "esp_min": float(np.random.uniform(-0.1, 0.05)),
            "status": "mock_success",
        }

    # Real calculation mode
    mol_raw = Chem.MolFromSmiles(smiles)
    if mol_raw is None: return None

    try:
        mol = get_best_conformer(mol_raw, num_confs=NUM_CONFS)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_name = os.path.join(tmpdir, "input.xyz")
            rdmolfiles.MolToXYZFile(mol, xyz_name)

            cmd = [
                XTB_EXE, os.path.basename(xyz_name),
                "--gfn", "2", "--opt", "--alpb", XTB_SOLVENT, "--vshift", "5"
            ]
            
            # Timeout prevents deadlocks
            proc = subprocess.run(
                cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, check=True, timeout=300
            )
            xtb_out = proc.stdout
            
            homo_ev, lumo_ev, gap_ev = parse_xtb_output(xtb_out)
            esp_min = parse_xtb_min_charge(xtb_out)

            if gap_ev is None: return None
            if esp_min is None: esp_min = 0.0

            return {
                "smiles": smiles,
                "EHOMO_ev": float(homo_ev),
                "ELUMO_ev": float(lumo_ev),
                "gap_ev": float(gap_ev),
                "esp_min": float(esp_min),
                "status": "xtb_success",
            }

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Step 3B: Parallel xTB Calculation")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Input file path (Step 3A output)")
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE, help="Output file path")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Process only the top K molecules")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel processes")
    parser.add_argument("--mock", action="store_true", help="Use mock data mode (for pipeline testing only)")
    args = parser.parse_args()

    # 1. Read input data
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return

    df = pd.read_csv(args.input_file)
    if "smiles" not in df.columns:
        print("❌ Input is missing the 'smiles' column")
        return

    # Sort and select top K
    if "Reward" in df.columns:
        df = df.sort_values("Reward", ascending=False).reset_index(drop=True)
    
    candidates = df.head(args.top_k).copy()
    print(f"🚀 [Step 3B] Starting xTB calculation: {len(candidates)} molecules, {args.workers} parallel workers")
    print(f"    Mode: {'MOCK (simulated)' if args.mock else 'REAL (actual calculation)'}")
    print(f"    Input: {args.input_file}")

    # 2. Prepare parallel tasks
    tasks = [(row["smiles"], i, args.mock) for i, (_, row) in enumerate(candidates.iterrows(), 1)]
    results = []
    
    # 3. Execute parallel calculation
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(run_xtb_task, tasks), 1):
            if res:
                results.append(res)
            
            # Progress reporting
            if i % 10 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\r⏳ Progress: {i}/{len(tasks)} | Success: {len(results)} | Elapsed: {elapsed:.1f}s | Speed: {speed:.2f} mol/s")
                sys.stdout.flush()

    print("\n")
    
    # 4. Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    cols = ["smiles", "EHOMO_ev", "ELUMO_ev", "gap_ev", "esp_min", "status"]
    
    if not results:
        print("⚠️ No valid results produced. Saving empty table.")
        pd.DataFrame(columns=cols).to_csv(args.output_file, index=False)
    else:
        df_res = pd.DataFrame(results)
        df_res = df_res[cols]  # Enforce column order
        df_res.to_csv(args.output_file, index=False)
        print(f"✅ xTB results saved to: {args.output_file}")
        print(f"📊 Success rate: {len(results)}/{len(candidates)} ({len(results)/len(candidates):.1%})")


if __name__ == "__main__":
    main()
