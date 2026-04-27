#!/bin/bash
# ─────────────────────────────────────────────
# Dataset Download Script
# Downloads Kaggle 5G Traffic and MAWI sample datasets.
# ─────────────────────────────────────────────

set -e

DATA_DIR="$(dirname "$0")/../data/raw"
mkdir -p "$DATA_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  FlowEmbed — Dataset Download Script     ║"
echo "╚══════════════════════════════════════════╝"

# ─────────────────────────────────────────────
# 1. Kaggle 5G Traffic Dataset
# ─────────────────────────────────────────────
echo ""
echo "📦 [1/3] Kaggle 5G Traffic Dataset"
echo "─────────────────────────────────────"

KAGGLE_DIR="$DATA_DIR/kaggle_5g"
mkdir -p "$KAGGLE_DIR"

if [ -f "$KAGGLE_DIR/5G_traffic.csv" ] || [ -n "$(ls -A "$KAGGLE_DIR" 2>/dev/null)" ]; then
    echo "  ✅ Already downloaded. Skipping."
else
    echo "  Download from: https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets"
    echo ""
    echo "  Option A (Kaggle CLI — if installed):"
    echo "    kaggle datasets download -d kimdaegyeom/5g-traffic-datasets -p $KAGGLE_DIR --unzip"
    echo ""
    echo "  Option B (Manual):"
    echo "    1. Visit the URL above"
    echo "    2. Click 'Download' (requires Kaggle account)"
    echo "    3. Extract to $KAGGLE_DIR"
    echo ""

    # Try Kaggle CLI if available
    if command -v kaggle &> /dev/null; then
        echo "  Kaggle CLI detected. Downloading..."
        kaggle datasets download -d kimdaegyeom/5g-traffic-datasets -p "$KAGGLE_DIR" --unzip
        echo "  ✅ Downloaded successfully."
    else
        echo "  ⚠️  Kaggle CLI not found. Please download manually."
        echo "  Install Kaggle CLI: pip install kaggle"
    fi
fi

# ─────────────────────────────────────────────
# 2. CESNET-QUIC22
# ─────────────────────────────────────────────
echo ""
echo "📦 [2/3] CESNET-QUIC22 Dataset"
echo "─────────────────────────────────────"

CESNET_DIR="$DATA_DIR/cesnet_quic22"
mkdir -p "$CESNET_DIR"

echo "  Download from: https://zenodo.org/records/7409924"
echo "  ⚠️  This dataset requires registration at cesnet.cz"
echo "  Download the flow statistics CSV (not full PCAPs unless needed)."
echo "  Place files in: $CESNET_DIR"

# ─────────────────────────────────────────────
# 3. Create Samsung S23 capture directory
# ─────────────────────────────────────────────
echo ""
echo "📱 [3/3] Samsung S23 Ultra Captures"
echo "─────────────────────────────────────"

S23_DIR="$DATA_DIR/samsung_s23"
mkdir -p "$S23_DIR"

echo "  Capture directory ready: $S23_DIR"
echo ""
echo "  📋 Capture Instructions:"
echo "  1. Connect S23 Ultra as USB tethering / WiFi hotspot"
echo "  2. Find interface: ifconfig (look for en* or bridge*)"
echo "  3. Start capture per app:"
echo "     sudo tcpdump -i <interface> -w $S23_DIR/youtube_4k.pcap"
echo "     (Run for 10-15 min per app)"
echo "  4. Apps to capture:"
echo "     - YouTube 4K streaming"
echo "     - Instagram Reels"
echo "     - BGMI / Mobile gaming"
echo "     - WhatsApp video call"
echo "     - Google Maps navigation"
echo "     - Background idle"
echo ""

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo "  📊 Dataset Status:"
echo "──────────────────────────────────────────"
echo "  Kaggle 5G:    $KAGGLE_DIR"
echo "  CESNET-QUIC:  $CESNET_DIR"
echo "  Samsung S23:  $S23_DIR"
echo "══════════════════════════════════════════"
