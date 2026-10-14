#!/bin/bash
# ==============================================================================
# ISL Prototype Setup Script for Raspberry Pi (Debian/Raspberry Pi OS 64-bit)
# ==============================================================================

set -e

echo "======================================================================"
echo "  🤟 Indian Sign Language (ISL) Wall Prototype Installer"
echo "======================================================================"

# 1. Update and install Linux system dependencies
echo "[1/5] Installing system audio, video, and build dependencies..."
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libatlas-base-dev \
    libportaudio2 \
    libasound2-dev \
    espeak \
    alsa-utils \
    ffmpeg \
    libgl1-mesa-glx

# 2. Python Virtual Environment
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Python Packages
echo "[3/5] Installing Python ML & Vision requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Audio Configuration (I2S DAC / MAX98357A check)
echo "[4/5] Checking I2S Audio DAC configuration..."
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

if [ -f "$CONFIG_FILE" ]; then
    if grep -q "dtoverlay=hifiberry-dac" "$CONFIG_FILE"; then
        echo "  ✓ I2S audio overlay (hifiberry-dac) already enabled in $CONFIG_FILE"
    else
        echo "  -> Adding 'dtoverlay=hifiberry-dac' to $CONFIG_FILE..."
        echo "dtoverlay=hifiberry-dac" | sudo tee -a "$CONFIG_FILE"
        echo "  ! Note: Reboot required for I2S audio driver to activate."
    fi
fi

# 5. Diagnostic Self-Check
echo "[5/5] Running diagnostic check..."
python diagnose_prototype.py --no-audio || true

echo "======================================================================"
echo "  ✅ Installation complete!"
echo "  To start the headless prototype manually:"
echo "      source venv/bin/activate"
echo "      python src/headless_prototype.py"
echo ""
echo "  To install as a power-on systemd service:"
echo "      sudo cp isl-detector.service /etc/systemd/system/"
echo "      sudo systemctl daemon-reload"
echo "      sudo systemctl enable isl-detector.service"
echo "      sudo systemctl start isl-detector.service"
echo "======================================================================"
