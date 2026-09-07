#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=========================================================="
echo "   Advanced ML Environment Setup & JupyterLab Launcher    "
echo "=========================================================="

# 1. Create .venv if not already created
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment in .venv..."
    python3 -m venv .venv
else
    echo "==> Found existing .venv."
fi

# 2. Activate virtual environment
echo "==> Activating .venv..."
source .venv/bin/activate

# 3. Register ipykernel for Jupyter notebooks
python -m ipykernel install --user --name aml-env --display-name "Python 3 (AML Course)" --quiet 2>/dev/null || true

# 4. Launch Jupyter Lab
echo "==> Launching JupyterLab with AML kernel available..."
exec jupyter lab
