#!/usr/bin/env bash
set -o errexit  # exit on first failure

echo "🧰 Forcing Python 3.10 installation"
apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-distutils

python3.10 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
