"""
Kaggle Notebook Script — Process CIC-AndMal2017 Android Malware Dataset
========================================================================

INSTRUCTIONS:
1. Open your Kaggle notebook with CICAndMal2017 dataset loaded.
2. Copy-paste this script into a cell and run it.
3. Download the output CSV from /kaggle/working/kaggle_andmal_flows.csv
4. Place it in your local Samsung/data/processed/ folder.

CIC-AndMal2017 contains CICFlowMeter-extracted CSVs organized by:
  - Adware-CSVs/
  - Benign-CSVs/
  - Ransomware-CSVs/
  - SMSmalware-CSVs/
  - Scareware-CSVs/

We map ALL malware categories to a single "malware" label.
We skip Benign (we already have plenty of benign training data).
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INPUT_DIR = "/kaggle/input"
OUTPUT_CSV = "/kaggle/working/kaggle_andmal_flows.csv"

# All malware types → "malware" label
# We skip "Benign" since we have plenty of benign traffic already
FOLDER_LABEL_MAP = {
    "adware": "malware",
    "ransomware": "malware",
    "smsmalware": "malware",
    "scareware": "malware",
    # "benign": "benign",  # Uncomment if you want benign too
}

# Maximum flows per malware category (to prevent class imbalance)
MAX_FLOWS_PER_CATEGORY = 5000


def get_label_from_path(filepath):
    """Determine label from the folder path."""
    path_lower = filepath.lower()
    for keyword, label in FOLDER_LABEL_MAP.items():
        if keyword in path_lower:
            return label
    return None


def map_cic_to_flowembed(df, label, source_file):
    """
    Map CICFlowMeter's 84 features to our 60-feature FlowEmbed schema.
    CIC uses different column names, so we map them carefully.
    """
    
    # CICFlowMeter column names can vary slightly between versions
    # We'll try multiple possible names for each feature
    def get_col(df, possible_names, default=0.0):
        for name in possible_names:
            # Try exact match first
            if name in df.columns:
                return df[name].fillna(default).replace([np.inf, -np.inf], default)
            # Try case-insensitive match
            for col in df.columns:
                if col.strip().lower() == name.strip().lower():
                    return df[col].fillna(default).replace([np.inf, -np.inf], default)
        return pd.Series([default] * len(df))

    result = pd.DataFrame()

    # ── Basic flow stats ──
    result["flow_duration_sec"] = get_col(df, ["Flow Duration", "flow_duration"]) / 1e6  # microseconds to seconds
    result["flow_duration_sec"] = result["flow_duration_sec"].clip(lower=0.001)
    
    fwd_pkts = get_col(df, ["Total Fwd Packets", "Tot Fwd Pkts", "total_fwd_packets"])
    bwd_pkts = get_col(df, ["Total Backward Packets", "Tot Bwd Pkts", "total_bwd_packets"])
    result["total_packets"] = fwd_pkts + bwd_pkts
    
    fwd_bytes = get_col(df, ["Total Length of Fwd Packets", "TotLen Fwd Pkts", "total_length_of_fwd_packets"])
    bwd_bytes = get_col(df, ["Total Length of Bwd Packets", "TotLen Bwd Pkts", "total_length_of_bwd_packets"])
    result["total_bytes"] = fwd_bytes + bwd_bytes
    
    result["packets_per_sec"] = get_col(df, ["Flow Packets/s", "Flow Pkts/s", "flow_pkts_s"])
    result["bytes_per_sec"] = get_col(df, ["Flow Bytes/s", "Flow Byts/s", "flow_byts_s"])

    # ── Packet size stats ──
    # CIC has Fwd and Bwd separately; we use the combined "Pkt Length" stats
    result["pkt_size_min"] = get_col(df, ["Min Packet Length", "Pkt Len Min", "min_packet_length"])
    result["pkt_size_max"] = get_col(df, ["Max Packet Length", "Pkt Len Max", "max_packet_length"])
    result["pkt_size_mean"] = get_col(df, ["Packet Length Mean", "Pkt Len Mean", "packet_length_mean"])
    result["pkt_size_std"] = get_col(df, ["Packet Length Std", "Pkt Len Std", "packet_length_std"])
    result["pkt_size_median"] = result["pkt_size_mean"]  # CIC doesn't have median, approximate
    result["pkt_size_q25"] = result["pkt_size_min"] + 0.25 * (result["pkt_size_max"] - result["pkt_size_min"])
    result["pkt_size_q75"] = result["pkt_size_min"] + 0.75 * (result["pkt_size_max"] - result["pkt_size_min"])
    result["pkt_size_iqr"] = result["pkt_size_q75"] - result["pkt_size_q25"]
    result["pkt_size_skewness"] = 0.0  # Not available in CIC
    result["pkt_size_kurtosis"] = 0.0  # Not available in CIC

    # ── IAT stats ──
    result["iat_min_ms"] = get_col(df, ["Flow IAT Min", "flow_iat_min"]) / 1000.0  # us to ms
    result["iat_max_ms"] = get_col(df, ["Flow IAT Max", "flow_iat_max"]) / 1000.0
    result["iat_mean_ms"] = get_col(df, ["Flow IAT Mean", "flow_iat_mean"]) / 1000.0
    result["iat_std_ms"] = get_col(df, ["Flow IAT Std", "flow_iat_std"]) / 1000.0
    result["iat_median_ms"] = result["iat_mean_ms"]  # Approximate
    result["iat_q25_ms"] = result["iat_min_ms"] + 0.25 * (result["iat_max_ms"] - result["iat_min_ms"])
    result["iat_q75_ms"] = result["iat_min_ms"] + 0.75 * (result["iat_max_ms"] - result["iat_min_ms"])

    # ── Protocol & encryption ──
    result["protocol_encoded"] = get_col(df, ["Protocol", "protocol"], default=6)
    dst_port = get_col(df, ["Destination Port", "Dst Port", "dst_port"], default=0)
    result["is_encrypted"] = (dst_port == 443).astype(int)

    # ── Entropy ──
    result["pkt_size_entropy"] = get_col(df, ["Packet Length Variance", "Pkt Len Var", "packet_length_variance"])
    # Normalize variance to approximate entropy (0-5 range)
    max_var = result["pkt_size_entropy"].max()
    if max_var > 0:
        result["pkt_size_entropy"] = (result["pkt_size_entropy"] / max_var) * 4.0
    result["iat_entropy"] = 0.0

    # ── Burst features (approximate from CIC features) ──
    # CIC has "Subflow" features which are similar to bursts
    subflow_fwd = get_col(df, ["Subflow Fwd Packets", "subflow_fwd_packets"], default=1)
    subflow_bwd = get_col(df, ["Subflow Bwd Packets", "subflow_bwd_packets"], default=1)
    
    result["burst_count"] = (subflow_fwd + subflow_bwd).clip(lower=1)
    result["burst_mean_size"] = result["total_packets"] / result["burst_count"]
    result["burst_mean_duration_ms"] = result["flow_duration_sec"] * 1000 / result["burst_count"]
    result["burst_rate"] = result["burst_count"] / result["flow_duration_sec"].clip(lower=0.001)
    result["inter_burst_mean_ms"] = result["flow_duration_sec"] * 1000 / result["burst_count"].clip(lower=1)
    result["burst_bytes_ratio"] = 0.8  # Approximation for malware (usually high)

    # ── First 20 packet sizes (approximate from available stats) ──
    for i in range(20):
        if i < 5:
            result[f"first_pkt_{i}"] = result["pkt_size_mean"]
        else:
            result[f"first_pkt_{i}"] = 0.0

    # ── Context features ──
    result["ctx_dst_ip_count"] = 1
    result["ctx_protocol_diversity"] = 1
    result["ctx_connection_rate"] = result["packets_per_sec"]
    result["ctx_packets_per_sec"] = result["packets_per_sec"]
    result["ctx_bytes_per_sec"] = result["bytes_per_sec"]

    # ── Ratio features ──
    result["avg_payload_ratio"] = result["pkt_size_mean"] / result["pkt_size_max"].clip(lower=1)
    result["iat_coefficient_of_variation"] = result["iat_std_ms"] / result["iat_mean_ms"].clip(lower=0.001)
    result["pkt_size_coefficient_of_variation"] = result["pkt_size_std"] / result["pkt_size_mean"].clip(lower=0.001)

    # ── Label and metadata ──
    result["label"] = label
    result["app_name"] = source_file

    # Clean up
    result = result.replace([np.inf, -np.inf], 0.0)
    result = result.fillna(0.0)

    return result


# ─────────────────────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────────────────────
def process_andmal():
    print(f"Searching for CSV files in {INPUT_DIR}...")

    csv_files = glob.glob(f"{INPUT_DIR}/**/*.csv", recursive=True)
    print(f"Found {len(csv_files)} CSV files total")

    if not csv_files:
        print("No CSV files found!")
        return

    # Filter and label
    labeled_files = []
    for f in csv_files:
        label = get_label_from_path(f)
        if label:
            labeled_files.append((f, label))

    print(f"Matched {len(labeled_files)} files to malware labels")

    # Show which folders we're processing
    folder_counts = Counter()
    for f, l in labeled_files:
        for keyword in FOLDER_LABEL_MAP:
            if keyword in f.lower():
                folder_counts[keyword] += 1
                break

    print(f"\nFiles per malware category:")
    for cat, count in sorted(folder_counts.items()):
        print(f"  {cat}: {count} files")

    all_dfs = []

    for i, (csv_path, label) in enumerate(labeled_files):
        try:
            filename = os.path.basename(csv_path)
            if i % 10 == 0:
                print(f"  Processing file {i+1}/{len(labeled_files)}: {filename}")

            df = pd.read_csv(csv_path, low_memory=False)

            # Skip empty files
            if len(df) == 0:
                continue

            # Print column names for first file (debugging)
            if i == 0:
                print(f"\n  CIC columns found: {df.columns.tolist()[:10]}...")
                print(f"  Total CIC columns: {len(df.columns)}")

            # Filter: at least 5 packets
            total_pkts_col = None
            for col_name in ["Total Fwd Packets", "Tot Fwd Pkts"]:
                if col_name in df.columns:
                    total_pkts_col = col_name
                    break

            if total_pkts_col:
                df = df[df[total_pkts_col] >= 3]

            if len(df) == 0:
                continue

            # Cap per file to prevent one massive file from dominating
            if len(df) > MAX_FLOWS_PER_CATEGORY:
                df = df.sample(n=MAX_FLOWS_PER_CATEGORY, random_state=42)

            # Map CIC features to our schema
            mapped = map_cic_to_flowembed(df, label, filename)
            all_dfs.append(mapped)

        except Exception as e:
            print(f"    ERROR processing {os.path.basename(csv_path)}: {e}")
            continue

    if not all_dfs:
        print("No flows extracted!")
        return

    result = pd.concat(all_dfs, ignore_index=True)

    # Final cap: max 15000 malware flows total (balanced with other classes)
    if len(result) > 15000:
        result = result.sample(n=15000, random_state=42)

    result.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"✅ CIC-AndMal2017 PROCESSING COMPLETE!")
    print(f"  Total malware flows: {len(result):,}")
    print(f"  Features per flow:   {len(result.columns) - 2}")
    print(f"  Output file:         {OUTPUT_CSV}")
    print(f"\n  Malware category distribution:")
    for app, cnt in result["app_name"].value_counts().head(10).items():
        print(f"    {app}: {cnt:,}")
    print(f"{'='*60}")
    print(f"\n📥 Download: {OUTPUT_CSV}")
    print(f"   Copy to local: Samsung/data/processed/kaggle_andmal_flows.csv")


if __name__ == "__main__":
    process_andmal()
