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
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(which python3)"
fi

"${PYTHON_BIN}" - << EOF
import os, shutil, glob, re

ROOT = os.path.abspath(".")
UPSTREAM = os.path.join(ROOT, "${UPSTREAM_DIR}")
NOTEBOOKS_DIR = os.path.join(ROOT, "04_Notebooks")

if not os.path.isdir(UPSTREAM):
    print("Upstream directory not found.")
    exit(0)

KNOWN_TOPICS = {
    8: "Polynomial_Regression",
    9: "Bias_Variance_Tradeoff",
    10: "Feature_Selection",
    11: "PCA_Dimensionality_Reduction"
}

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
        # Infer meaningful topic name
        topic = KNOWN_TOPICS.get(lab_num, "")
        if not topic:
            # Check subdirectories
            for sub in os.listdir(lab_src):
                if os.path.isdir(os.path.join(lab_src, sub)) and "solved" not in sub.lower():
                    clean_sub = re.sub(r"^(Lab_|L)\d+_", "", sub, flags=re.IGNORECASE)
                    clean_sub = re.sub(r"_(student|todo|copy)", "", clean_sub, flags=re.IGNORECASE)
                    if clean_sub:
                        topic = clean_sub.title().replace(" ", "_")
                        break
        if not topic:
            # Check notebook files in lab_src
            for root_w, _, files_w in os.walk(lab_src):
                for fw in files_w:
                    if fw.endswith(".ipynb") and not fw.startswith(".") and "solved" not in fw.lower() and "instructor" not in fw.lower():
                        clean_fn = re.sub(r"^(Lab_|L)\d+_", "", fw.replace(".ipynb", ""), flags=re.IGNORECASE)
                        clean_fn = re.sub(r"_(student|todo|copy|final)", "", clean_fn, flags=re.IGNORECASE)
                        clean_fn = re.sub(r"^(student|todo)_", "", clean_fn, flags=re.IGNORECASE)
                        if clean_fn:
                            topic = clean_fn.title().replace(" ", "_")
                            break
                if topic:
                    break
        if not topic:
            topic = f"Lab_{lab_num}"

        target_lab_dir = os.path.join(NOTEBOOKS_DIR, f"{lab_str}_{topic}")
        os.makedirs(target_lab_dir, exist_ok=True)
        print(f"==> Created new lab folder: {os.path.relpath(target_lab_dir, ROOT)}")

    raw_dir = os.path.join(target_lab_dir, "raw")
    solved_dir = os.path.join(target_lab_dir, "solved")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(solved_dir, exist_ok=True)
    os.makedirs(os.path.join(solved_dir, ".ipynb_checkpoints"), exist_ok=True)

    # Locate raw template directory in upstream (e.g. Lab_X_Student_copy, Lab_X_Student_todo, or non-solved dir, or lab root)
    subdirs = [d for d in os.listdir(lab_src) if os.path.isdir(os.path.join(lab_src, d)) and "solved" not in d.lower()]
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

print("\n==> Sync check complete.")
EOF

echo "=========================================================="
echo "    Lab synchronization completed successfully!           "
echo "=========================================================="
