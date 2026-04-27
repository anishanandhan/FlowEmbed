"""
Kaggle Notebook Script — Process CESNET QUIC 2022 Dataset
=========================================================

INSTRUCTIONS:
1. Open your Kaggle notebook with the CESNET dataset loaded.
2. Copy-paste this entire script into a code cell.
3. Run the cell.
4. Download the output CSV from /kaggle/working/kaggle_cesnet_flows.csv
5. Place it in your local `data/processed/` folder.

This script parses the CESNET flows-*.csv files, maps the SNI/App labels
to our 4 target classes (gaming, streaming, voip, xr_ar), and reconstructs
the exact 60 features expected by FlowEmbed (by parsing the SIZE_SEQ and 
TIME_SEQ arrays if available).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ast

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INPUT_DIR = Path("/kaggle/input/datasets/anishanandhan/cesnet/cesnet-quic22")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_CSV = OUTPUT_DIR / "kaggle_cesnet_flows.csv"

# Adjust this to process more/fewer files if you run out of RAM
MAX_FILES_TO_PROCESS = 5  
MAX_ROWS_PER_FILE = 200000

# Map CESNET SNI or App categories to our 4 target classes
# You may need to adjust these keywords based on the actual CESNET SNIs
CATEGORY_KEYWORDS = {
    "gaming": ["playstation", "xbox", "steam", "blizzard", "epicgames", "ea.com", "riotgames"],
    "streaming": ["youtube", "netflix", "nflxvideo", "video", "twitch", "disney", "hulu", "vimeo", "prime"],
    "voip": ["zoom.us", "teams.microsoft", "meet.google", "skype", "discordapp", "webex", "chime"],
    "xr_ar": ["oculus", "meta", "roblox", "vr", "ar"]
}

def get_label_from_sni(sni: str) -> str:
    if not isinstance(sni, str):
        return None
    sni = sni.lower()
    for label, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in sni for kw in keywords):
            return label
    return None


# ─────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────
def compute_entropy(values: np.ndarray, num_bins: int = 20) -> float:
    if len(values) < 2: return 0.0
    try:
        hist, _ = np.histogram(values, bins=num_bins, density=False)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    except: return 0.0

def detect_bursts(iats_ms: np.ndarray, lengths: np.ndarray, threshold_ms: float = 50.0):
    empty = {
        "burst_count": 0, "burst_mean_size": 0.0, "burst_mean_duration_ms": 0.0,
        "burst_rate": 0.0, "inter_burst_mean_ms": 0.0, "burst_bytes_ratio": 0.0,
    }
    if len(iats_ms) < 2: return empty

    bursts, start, pkt_count = [], 0, 1
    for i, iat in enumerate(iats_ms):
        if iat < threshold_ms:
            pkt_count += 1
        else:
            if pkt_count >= 2:
                bursts.append({
                    "packets": pkt_count, 
                    "bytes": float(lengths[start:start+pkt_count].sum()),
                    "duration_ms": float(iats_ms[start:start+pkt_count-1].sum() if len(iats_ms[start:start+pkt_count-1])>0 else 0)
                })
            start, pkt_count = i + 1, 1

    if pkt_count >= 2:
        bursts.append({
            "packets": pkt_count, 
            "bytes": float(lengths[start:start+pkt_count].sum()),
            "duration_ms": float(iats_ms[start:start+pkt_count-1].sum() if len(iats_ms[start:start+pkt_count-1])>0 else 0)
        })

    if not bursts: return empty

    total_dur = max(float(iats_ms.sum()), 1.0)
    burst_bytes = sum(b["bytes"] for b in bursts)
    burst_durs = [b["duration_ms"] for b in bursts]

    return {
        "burst_count": len(bursts),
        "burst_mean_size": np.mean([b["packets"] for b in bursts]),
        "burst_mean_duration_ms": np.mean(burst_durs),
        "burst_rate": len(bursts) / max(total_dur / 1000, 0.001),
        "inter_burst_mean_ms": (total_dur - sum(burst_durs)) / max(len(bursts), 1),
        "burst_bytes_ratio": burst_bytes / max(float(lengths.sum()), 1),
    }

def process_cesnet_row(row, label: str):
    """
    Reconstruct our 60 features from the CESNET aggregated flow row.
    Assumes presence of SIZE_SEQ and TIME_SEQ arrays for packet-level details.
    """
    try:
        # Parse packet size and IAT sequences (CESNET provides these as strings of lists)
        sizes = np.array(ast.literal_eval(row.get("SIZE_SEQ", "[]")) if isinstance(row.get("SIZE_SEQ"), str) else row.get("SIZE_SEQ", []))
        times = np.array(ast.literal_eval(row.get("TIME_SEQ", "[]")) if isinstance(row.get("TIME_SEQ"), str) else row.get("TIME_SEQ", []))
        
        # If lengths aren't arrays but we have overall stats, we pad
        if len(sizes) == 0:
            sizes = np.array([float(row.get("IN_BYTES", 0) + row.get("OUT_BYTES", 0)) / max(1, row.get("IN_PKTS", 1) + row.get("OUT_PKTS", 1))] * max(1, row.get("IN_PKTS", 0)))
        if len(times) == 0:
            times = np.zeros(len(sizes)-1) if len(sizes) > 1 else np.array([])
            
        dur_sec = max(float(row.get("FLOW_DURATION_MILLISECONDS", row.get("DURATION", 1000))) / 1000.0, 0.001)
        
        f = {}
        # Basic stats
        f["flow_duration_sec"] = dur_sec
        f["total_packets"] = len(sizes)
        f["total_bytes"] = float(sizes.sum())
        f["packets_per_sec"] = len(sizes) / dur_sec
        f["bytes_per_sec"] = float(sizes.sum()) / dur_sec

        # Packet size stats
        f["pkt_size_min"] = float(sizes.min()) if len(sizes) > 0 else 0.0
        f["pkt_size_max"] = float(sizes.max()) if len(sizes) > 0 else 0.0
        f["pkt_size_mean"] = float(sizes.mean()) if len(sizes) > 0 else 0.0
        f["pkt_size_std"] = float(sizes.std()) if len(sizes) > 1 else 0.0
        f["pkt_size_median"] = float(np.median(sizes)) if len(sizes) > 0 else 0.0
        f["pkt_size_q25"] = float(np.percentile(sizes, 25)) if len(sizes) > 0 else 0.0
        f["pkt_size_q75"] = float(np.percentile(sizes, 75)) if len(sizes) > 0 else 0.0
        f["pkt_size_iqr"] = f["pkt_size_q75"] - f["pkt_size_q25"]
        f["pkt_size_skewness"] = 0.0  # Simplified for speed
        f["pkt_size_kurtosis"] = 0.0

        # IAT stats
        f["iat_min_ms"] = float(times.min()) if len(times) > 0 else 0.0
        f["iat_max_ms"] = float(times.max()) if len(times) > 0 else 0.0
        f["iat_mean_ms"] = float(times.mean()) if len(times) > 0 else 0.0
        f["iat_std_ms"] = float(times.std()) if len(times) > 1 else 0.0
        f["iat_median_ms"] = float(np.median(times)) if len(times) > 0 else 0.0
        f["iat_q25_ms"] = float(np.percentile(times, 25)) if len(times) > 0 else 0.0
        f["iat_q75_ms"] = float(np.percentile(times, 75)) if len(times) > 0 else 0.0

        # Encodings
        f["protocol_encoded"] = 2  # CESNET-QUIC is 100% QUIC
        f["is_encrypted"] = 1.0    # QUIC is encrypted

        # Entropies
        f["pkt_size_entropy"] = compute_entropy(sizes)
        f["iat_entropy"] = compute_entropy(times[times > 0]) if len(times) > 0 else 0.0

        # Bursts
        f.update(detect_bursts(times, sizes))

        # First 20 packets (Zero-padded if flow is too short)
        for i in range(20):
            f[f"first_pkt_{i}"] = float(sizes[i]) if i < len(sizes) else 0.0

        # Context features (Mocked if we don't do complex grouping to save RAM)
        f["ctx_dst_ip_count"] = 1
        f["ctx_protocol_diversity"] = 1
        f["ctx_connection_rate"] = 0.0
        f["ctx_packets_per_sec"] = f["packets_per_sec"]
        f["ctx_bytes_per_sec"] = f["bytes_per_sec"]

        # Ratios
        f["avg_payload_ratio"] = f["pkt_size_mean"] / max(f["pkt_size_max"], 1)
        f["iat_coefficient_of_variation"] = f["iat_std_ms"] / max(f["iat_mean_ms"], 0.001)
        f["pkt_size_coefficient_of_variation"] = f["pkt_size_std"] / max(f["pkt_size_mean"], 0.001)

        f["label"] = label
        return f
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════════╗")
print("║  FlowEmbed — CESNET QUIC 2022 Flow Feature Extraction    ║")
print("╚══════════════════════════════════════════════════════════╝\n")

csv_files = list(INPUT_DIR.rglob("flows-*.csv"))
if not csv_files:
    print("❌ No flow CSV files found in", INPUT_DIR)
else:
    print(f"Found {len(csv_files)} CSV files. Processing up to {MAX_FILES_TO_PROCESS}...")

all_processed_flows = []
files_processed = 0

for file_path in csv_files:
    if files_processed >= MAX_FILES_TO_PROCESS:
        break
        
    print(f"\nProcessing {file_path.name}...")
    try:
        # Read a chunk of rows to save RAM
        df = pd.read_csv(file_path, nrows=MAX_ROWS_PER_FILE, low_memory=False)
        print(f"  Loaded {len(df)} rows. Mapping labels...")
        
        # Determine the SNI column (usually SNI, APP, or TLS_SNI)
        sni_col = next((c for c in df.columns if c.upper() in ["SNI", "APP", "TLS_SNI", "SERVER_NAME"]), None)
        
        if not sni_col:
            print(f"  ⚠ Skipping: Could not find SNI/APP column in {list(df.columns[:5])}...")
            continue
            
        # Map SNI to our 4 labels
        df["mapped_label"] = df[sni_col].apply(get_label_from_sni)
        
        # Filter only to rows that matched a label
        df_filtered = df[df["mapped_label"].notna()]
        print(f"  Found {len(df_filtered)} labeled flows.")
        
        if len(df_filtered) == 0:
            continue
            
        # Extract features
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="  Extracting"):
            features = process_cesnet_row(row, row["mapped_label"])
            if features:
                all_processed_flows.append(features)
                
        files_processed += 1
        
    except Exception as e:
        print(f"  ERROR processing {file_path.name}: {e}")

if all_processed_flows:
    final_df = pd.DataFrame(all_processed_flows)
    
    # Save output
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"  Total flows extracted: {len(final_df):,}")
    print(f"  Output file: {OUTPUT_CSV}")
    print(f"  File size: {OUTPUT_CSV.stat().st_size / (1024*1024):.1f} MB")
    print(f"\n  Class distribution:")
    for lbl, cnt in final_df["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,} flows")
    print(f"{'='*60}")
    print(f"\n📥 Download: {OUTPUT_CSV}")
    print(f"   Copy this to your local: Samsung/data/processed/cesnet_flows.csv")
else:
    print("\n❌ No flows extracted. Check the SNI column names or CATEGORY_KEYWORDS.")
