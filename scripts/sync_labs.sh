#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=========================================================="
echo "    Syncing Labs from Upstream (gourabrajak-cloud/aml_lab)  "
echo "=========================================================="

# 1. Determine upstream location (hidden in .upstream/aml_lab)
UPSTREAM_DIR=".upstream/aml_lab"
if [ ! -d "${UPSTREAM_DIR}/.git" ]; then
    echo "==> Cloning upstream repository https://github.com/gourabrajak-cloud/aml_lab.git..."
    git clone https://github.com/gourabrajak-cloud/aml_lab.git "${UPSTREAM_DIR}"
else
    echo "==> Pulling latest updates in ${UPSTREAM_DIR}..."
    git -C "${UPSTREAM_DIR}" pull origin main
fi

# 2. Run sync logic via Python
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

    raw_dir = os.path.join(target_lab_dir, "raw")
    solved_dir = os.path.join(target_lab_dir, "solved")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(solved_dir, exist_ok=True)
    os.makedirs(os.path.join(solved_dir, ".ipynb_checkpoints"), exist_ok=True)

    # Locate raw template directory in upstream (e.g. Lab_X_Student_copy, Lab_X_Student_todo, or lab root)
    subdirs = [d for d in os.listdir(lab_src) if os.path.isdir(os.path.join(lab_src, d)) and "student" in d.lower()]
    raw_source_dir = os.path.join(lab_src, subdirs[0]) if subdirs else lab_src

    # 1. Update raw/ with untouched files from upstream
    for f in os.listdir(raw_source_dir):
        src_file = os.path.join(raw_source_dir, f)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(raw_dir, f))

    # 2. If solved/ is empty, initialize it with the starter files ready to work on
    existing_solved_files = [f for f in os.listdir(solved_dir) if not f.startswith(".")]
    if not existing_solved_files:
        for f in os.listdir(raw_dir):
            src_file = os.path.join(raw_dir, f)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(solved_dir, f))
        print(f"    [Initialized solved/] for {os.path.basename(target_lab_dir)}")

    # 3. Check for teacher solved reference and copy if available
    for item in os.listdir(lab_src):
        if "solved" in item.lower():
            solved_source_dir = os.path.join(lab_src, item)
            if os.path.isdir(solved_source_dir):
                for sf in os.listdir(solved_source_dir):
                    if sf.endswith(".ipynb"):
                        ref_dst = os.path.join(solved_dir, "Teacher_Reference_Solved.ipynb")
                        if not os.path.exists(ref_dst):
                            shutil.copy2(os.path.join(solved_source_dir, sf), ref_dst)
                            print(f"    [Teacher reference added] Teacher_Reference_Solved.ipynb")

print("\n==> Sync check complete.")
EOF

echo "=========================================================="
echo "    Lab synchronization completed successfully!           "
echo "=========================================================="
