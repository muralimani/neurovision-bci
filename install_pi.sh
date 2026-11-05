#!/usr/bin/env bash
# One-time setup on Raspberry Pi OS (64-bit Bookworm or later recommended).
# Usage:  bash install_pi.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libopenblas0 libatlas-base-dev || true

echo "==> Python virtual environment (.venv)"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "==> PiEEG server (needed once the board arrives; harmless to install now)"
pip install pieeg-server || echo "pieeg-server install failed - retry later with: pip install pieeg-server"

echo "==> Synthetic datasets"
python -m neurovision generate-data

cat <<'EOF'

Done. Next steps:
  source .venv/bin/activate
  export GEMINI_API_KEY="your-key-from-aistudio.google.com"

  # test without any API calls
  python -m neurovision run --scenario "relaxed:40,focused:40,stressed:30" --dry-run

  # one real image
  python -m neurovision once --state meditative

The first time pieeg-server runs it enables SPI; reboot once afterwards (sudo reboot).
EOF
