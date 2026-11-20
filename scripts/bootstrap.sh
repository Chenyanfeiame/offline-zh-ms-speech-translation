#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
VENV_DIR="${VENV_DIR:-.venv}"
"${PYTHON_BIN}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required"'
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -c constraints-tested.txt -e '.[models,test]'
cp -n config.example.json config.json || true

echo "Environment ready. Next: ${VENV_DIR}/bin/python scripts/prepare_models.py"
