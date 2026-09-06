"""
Smoke-test: convert first 5 muatm MC reco ROOT files to a small HDF5,
then verify that raw/labels in H5 matches fFlag from ROOT (GT ground truth).

Run from data_manager/root2h5/:
    python test_mc_reco_small.py
"""

import os
import subprocess
import numpy as np
import uproot
import h5py
import yaml
import shutil

# ── config ────────────────────────────────────────────────────────────────────
ROOT_DIR   = "/home/albert/Baikal2025/data_manager/data/mc_reco_root/muatm_files"
H5_OUT_DIR = "/home/albert/Baikal2025/data_manager/data/h5datasets"
H5_NAME    = "test_mc_reco_small.h5"
H5_OUT     = os.path.join(H5_OUT_DIR, H5_NAME)
N_FILES    = 5

FLAG_BRANCH  = ("Events/MCEventMask./MCEventMask.BEventMask"
                "/MCEventMask.BEventMask.fOrigins/MCEventMask.BEventMask.fOrigins.fFlag")
NHITS_BRANCH = "Events/BRecoMuon./BRecoMuon.fNHits"
PULSE_N      = "Events/BEvent./BEvent.fPulseN"

# ── pick first N_FILES root files ─────────────────────────────────────────────
root_files = sorted(f for f in os.listdir(ROOT_DIR) if f.endswith(".root"))[:N_FILES]
print(f"Will convert {len(root_files)} files:")
for f in root_files:
    print(f"  {f}")

# ── write a limited-files-only root dir via symlinks in a temp dir ─────────────
TEMP_ROOT_DIR = "/tmp/test_mc_reco_root_subset"
if os.path.exists(TEMP_ROOT_DIR):
    shutil.rmtree(TEMP_ROOT_DIR)
os.makedirs(TEMP_ROOT_DIR)
for f in root_files:
    os.symlink(os.path.join(ROOT_DIR, f), os.path.join(TEMP_ROOT_DIR, f))

# ── write patched config ───────────────────────────────────────────────────────
BASE_CFG = "root2h5_config_mc_reco.yaml"
TEST_CFG = "root2h5_config_mc_reco_test.yaml"

with open(BASE_CFG) as fh:
    cfg = yaml.safe_load(fh)

cfg["input"]  = {"particle": "muatm", "root_dir_path": TEMP_ROOT_DIR}
cfg["output"] = {"h5_name": H5_NAME, "h5_prefix": H5_OUT_DIR}
cfg["multiprocessing"] = {"MAX_QUEUE_SIZE": 3, "NUM_WORKERS": 2}

with open(TEST_CFG, "w") as fh:
    yaml.dump(cfg, fh)

print(f"\nPatched config → {TEST_CFG}")

# ── patch the converter to read the test config, run as subprocess ─────────────
CONVERTER = "root2h5_mc_reco.py"
PATCHED   = "root2h5_mc_reco_test_run.py"

with open(CONVERTER) as fh:
    src = fh.read()

src_patched = src.replace(
    "read_config('root2h5_config_mc_reco.yaml')",
    f"read_config('{TEST_CFG}')",
)

with open(PATCHED, "w") as fh:
    fh.write(src_patched)

if os.path.exists(H5_OUT):
    os.remove(H5_OUT)

print(f"Running converter (patched) → {PATCHED} ...")
result = subprocess.run(["python", PATCHED], capture_output=False)
if result.returncode != 0:
    print("Converter exited with error.")
    raise SystemExit(1)

# ── cleanup temp files ────────────────────────────────────────────────────────
os.remove(PATCHED)
os.remove(TEST_CFG)
shutil.rmtree(TEMP_ROOT_DIR)

print(f"\nConverter done. H5: {H5_OUT}")

# ── verification ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERIFICATION: H5 raw/labels  vs  ROOT fFlag")
print("=" * 70)

with h5py.File(H5_OUT, "r") as hf:
    parts = sorted(hf["muatm/reco_prty"].keys())
    print(f"Parts in H5: {len(parts)}\n")

    total_root_both = 0
    total_h5_both   = 0

    for part in parts:
        part_name = part[len("part_"):]
        root_path = os.path.join(ROOT_DIR, part_name + ".root")

        h5_labels    = hf[f"muatm/raw/labels/{part}/data"][:]
        h5_ev_starts = hf[f"muatm/raw/ev_starts/{part}/data"][:]
        h5_nhits     = hf[f"muatm/reco_prty/{part}/data"][:, 8]

        n_ev_h5 = len(h5_nhits)
        starts  = h5_ev_starts[:-1].astype(np.intp)
        h5_gt   = np.add.reduceat((h5_labels != 0).astype(np.int32), starts)

        with uproot.open(root_path) as rf:
            pulse_n    = rf[PULSE_N].array(library="np")
            st         = 1 if pulse_n[0] == 0 else 0
            root_flags = rf[FLAG_BRANCH].array(library="np")[st:]
            root_nhits = rf[NHITS_BRANCH].array(library="np")[st:]

        n_ev_root   = len(root_nhits)
        root_gt     = np.array([int(np.sum(ev != 0)) for ev in root_flags])
        root_flat   = np.concatenate([ev for ev in root_flags])

        root_both = int(((root_gt == 0) & (root_nhits >= 8)).sum())
        h5_both   = int(((h5_gt   == 0) & (h5_nhits   >= 8)).sum())
        total_root_both += root_both
        total_h5_both   += h5_both

        # Label match check (only meaningful when sizes agree)
        if len(root_flat) == len(h5_labels):
            label_match = "MATCH ✓" if np.array_equal(root_flat, h5_labels) else "MISMATCH ✗"
        else:
            label_match = f"size differs (ROOT flat {len(root_flat)} vs H5 {len(h5_labels)}) — cluster filtering/expansion"

        # Reco mask fields
        has_reco_mask = f"muatm/raw/reco_mask/{part}/data" in hf
        if has_reco_mask:
            reco_mask_flat  = hf[f"muatm/raw/reco_mask/{part}/data"][:]
            reco_n_sig_hits = hf[f"muatm/raw/reco_n_sig_hits/{part}/data"][:]
            reco_gt0 = int((reco_n_sig_hits == 0).sum())
            reco_gt8 = int((reco_n_sig_hits >= 8).sum())
            reco_mask_info = (f"reco_n_sig_hits: min={reco_n_sig_hits.min()} "
                              f"max={reco_n_sig_hits.max()} | "
                              f"reco_sig==0: {reco_gt0} | reco_sig>=8: {reco_gt8}")
        else:
            reco_mask_info = "reco_mask NOT FOUND"

        print(f"  {part_name}")
        print(f"    ROOT: {n_ev_root} events | GT==0: {int((root_gt==0).sum()):4d} | nHits>=8: {int((root_nhits>=8).sum()):4d} | BOTH: {root_both}")
        print(f"    H5  : {n_ev_h5} events | GT==0: {int((h5_gt==0).sum()):4d} | nHits>=8: {int((h5_nhits>=8).sum()):4d} | BOTH: {h5_both}")
        print(f"    labels: {label_match}")
        print(f"    {reco_mask_info}")
        print()

    print(f"TOTAL  ROOT both: {total_root_both}")
    print(f"TOTAL  H5   both: {total_h5_both}")
