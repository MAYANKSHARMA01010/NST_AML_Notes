#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=========================================================="
echo "    Syncing Labs from Upstream (gourabrajak-cloud/aml_lab)  "
echo "=========================================================="

# 1. Determine upstream location (support hidden .upstream/aml_lab)
if [ -d ".upstream/aml_lab/.git" ]; then
    UPSTREAM_DIR=".upstream/aml_lab"
elif [ -d "aml_lab/.git" ]; then
    UPSTREAM_DIR="aml_lab"
else
    UPSTREAM_DIR=".upstream/aml_lab"
    echo "==> Cloning upstream repository https://github.com/gourabrajak-cloud/aml_lab.git..."
    git clone https://github.com/gourabrajak-cloud/aml_lab.git "${UPSTREAM_DIR}"
fi

echo "==> Pulling latest updates in ${UPSTREAM_DIR}..."
git -C "${UPSTREAM_DIR}" pull origin main

# 2. Run clean sync logic via Python
"${ROOT_DIR}/.venv/bin/python" - << EOF
import os, shutil, glob, re

ROOT = os.path.abspath(".")
UPSTREAM = os.path.join(ROOT, "${UPSTREAM_DIR}")
NOTEBOOKS_DIR = os.path.join(ROOT, "04_Notebooks")

if not os.path.isdir(UPSTREAM):
    print("Upstream directory not found.")
    exit(0)

# Scan for Lab directories in upstream (e.g. Lab_3, Lab_4, Lab_5, Lab_6, Lab_7, Lab_8, etc.)
lab_dirs = sorted([d for d in os.listdir(UPSTREAM) if re.match(r"^Lab_\d+", d, re.IGNORECASE)])

print(f"==> Scanning upstream labs: {', '.join(lab_dirs)}")

for lab in lab_dirs:
    lab_src = os.path.join(UPSTREAM, lab)
    if not os.path.isdir(lab_src):
        continue
    
    # Extract lab number
    m = re.search(r"Lab_(\d+)", lab, re.IGNORECASE)
    lab_num = int(m.group(1)) if m else 0
    lab_str = f"Lab_{lab_num:02d}"

    # Search for matching directory in 04_Notebooks
    matches = [d for d in os.listdir(NOTEBOOKS_DIR) if d.startswith(lab_str) or d.startswith(f"Lab_{lab_num}_")]
    if matches:
        target_lab_dir = os.path.join(NOTEBOOKS_DIR, matches[0])
    else:
        lab_name_clean = lab.replace("_", " ").title().replace(" ", "_")
        target_lab_dir = os.path.join(NOTEBOOKS_DIR, f"{lab_str}_{lab_name_clean}")
        os.makedirs(target_lab_dir, exist_ok=True)
        print(f"==> Created new lab folder: {os.path.relpath(target_lab_dir, ROOT)}")

    # Copy lab-specific requirements.txt if present and not already existing
    req_src = os.path.join(lab_src, "requirements.txt")
    req_dst = os.path.join(target_lab_dir, "requirements.txt")
    if os.path.exists(req_src) and not os.path.exists(req_dst):
        shutil.copy2(req_src, req_dst)
        print(f"    [Requirements added] requirements.txt in {os.path.relpath(target_lab_dir, ROOT)}")

    # Copy datasets, text prompts, diagrams into target lab directory directly (NO subdirectories)
    for ext in ["*.csv", "*.txt", "*.png", "*.jpg", "*.svg"]:
        for asset in glob.glob(os.path.join(lab_src, "**/" + ext), recursive=True):
            fname = os.path.basename(asset)
            dst = os.path.join(target_lab_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(asset, dst)
                print(f"    [Asset copied] {fname} -> {os.path.relpath(target_lab_dir, ROOT)}")

    # Copy newly assigned starter notebook if no notebook exists in target_lab_dir
    existing_nbs = [f for f in os.listdir(target_lab_dir) if f.endswith(".ipynb")]
    if not existing_nbs:
        for nb in glob.glob(os.path.join(lab_src, "*.ipynb")):
            fname = os.path.basename(nb)
            dst_nb = os.path.join(target_lab_dir, fname)
            shutil.copy2(nb, dst_nb)
            print(f"    [New assignment notebook copied] {fname} -> {os.path.relpath(target_lab_dir, ROOT)}")

    # Check for solved reference notebook and copy directly with _Solved suffix if missing
    for item in os.listdir(lab_src):
        if "solved" in item.lower():
            solved_src = os.path.join(lab_src, item)
            if os.path.isdir(solved_src):
                for sf in os.listdir(solved_src):
                    if sf.endswith(".ipynb"):
                        solved_name = f"{lab_str}_Solved_{sf}" if "solved" not in sf.lower() else sf
                        dst_solved = os.path.join(target_lab_dir, solved_name)
                        if not os.path.exists(dst_solved) and not any("solved" in existing.lower() for existing in existing_nbs):
                            shutil.copy2(os.path.join(solved_src, sf), dst_solved)
                            print(f"    [Reference solution copied] {solved_name}")

print("\n==> Sync check complete.")
EOF

echo "=========================================================="
echo "    Lab synchronization completed successfully!           "
echo "=========================================================="
