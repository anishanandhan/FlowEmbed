"""
Kaggle Notebook Script — Process 5G Traffic Dataset on Kaggle Cloud
====================================================================

INSTRUCTIONS:
1. Go to https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets
2. Click "New Notebook" 
3. Copy-paste this entire script into a code cell
4. Click "Run All"
5. Download the output CSV from /kaggle/working/kaggle_5g_flows.csv
   (tiny ~5-10 MB file)

The 45GB raw data stays on Kaggle servers — you only download the 
processed flow CSV with ~56 features per flow.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INPUT_DIR = Path("/kaggle/input/5g-traffic-datasets/5G_Traffic_Datasets")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_CSV = OUTPUT_DIR / "kaggle_5g_flows.csv"

FIRST_N_PACKETS = 20        # Handshake fingerprint sequence length
BURST_THRESHOLD_MS = 50     # Burst detection IAT threshold
MAX_PACKETS_PER_FILE = 500000  # Cap per CSV to manage RAM
MAX_FILES_PER_APP = 10      # Process all files (Kaggle has enough RAM)

# Category → label mapping
CATEGORY_LABEL_MAP = {
    "Game_Streaming": "gaming",
    "Online_Game": "gaming",
    "Stored_Streaming": "streaming",
    "Live_Streaming": "streaming",
    "Video_Conferencing": "voip",
    "Metaverse": "xr_ar",
}


# ─────────────────────────────────────────────────────────────
# Feature Extraction Functions
# ─────────────────────────────────────────────────────────────

def compute_entropy(values: np.ndarray, num_bins: int = 20) -> float:
    """Shannon entropy of a distribution."""
    if len(values) < 2:
        return 0.0
    try:
        hist, _ = np.histogram(values, bins=num_bins, density=False)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    except Exception:
        return 0.0


def detect_bursts(iats_ms: np.ndarray, lengths: np.ndarray, threshold_ms: float = BURST_THRESHOLD_MS) -> Dict:
    """Detect burst patterns in traffic."""
    empty = {
        "burst_count": 0, "burst_mean_size": 0.0, "burst_mean_duration_ms": 0.0,
        "burst_rate": 0.0, "inter_burst_mean_ms": 0.0, "burst_bytes_ratio": 0.0,
    }
    if len(iats_ms) < 2:
        return empty

    bursts = []
    start, pkt_count = 0, 1

    for i, iat in enumerate(iats_ms):
        if iat < threshold_ms:
            pkt_count += 1
        else:
            if pkt_count >= 2:
                bl = lengths[start:start + pkt_count]
                bi = iats_ms[start:start + pkt_count - 1]
                bursts.append({"packets": pkt_count, "bytes": float(bl.sum()),
                               "duration_ms": float(bi.sum()) if len(bi) > 0 else 0.0})
            start, pkt_count = i + 1, 1

    if pkt_count >= 2:
        bl = lengths[start:start + pkt_count]
        bi = iats_ms[start:start + pkt_count - 1]
        bursts.append({"packets": pkt_count, "bytes": float(bl.sum()),
                       "duration_ms": float(bi.sum()) if len(bi) > 0 else 0.0})

    if not bursts:
        return empty

    total_bytes = float(lengths.sum())
    total_dur = float(iats_ms.sum()) if len(iats_ms) > 0 else 1.0
    burst_bytes = sum(b["bytes"] for b in bursts)
    burst_durs = [b["duration_ms"] for b in bursts]

    return {
        "burst_count": len(bursts),
        "burst_mean_size": np.mean([b["packets"] for b in bursts]),
        "burst_mean_duration_ms": np.mean(burst_durs),
        "burst_rate": len(bursts) / max(total_dur / 1000, 0.001),
        "inter_burst_mean_ms": (total_dur - sum(burst_durs)) / max(len(bursts), 1),
        "burst_bytes_ratio": burst_bytes / max(total_bytes, 1),
    }


def compute_source_ip_context(packets: pd.DataFrame) -> Dict:
    """Context-aware host features (FlowXpert-inspired)."""
    context = {}
    for src_ip, group in packets.groupby("Source"):
        ts = group["timestamp"]
        dur = max((ts.max() - ts.min()).total_seconds(), 0.001)
        context[src_ip] = {
            "src_dst_ip_count": group["Destination"].nunique(),
            "src_protocol_diversity": group["Protocol"].nunique(),
            "src_flow_count": len(group.groupby(["Destination", "Protocol"])),
            "src_connection_rate": len(group.groupby(["Destination", "Protocol"])) / dur,
            "src_packets_per_sec": len(group) / dur,
            "src_bytes_per_sec": group["Length"].sum() / dur,
        }
    return context


def extract_flow_features(packets, src, dst, protocol, label, app_name, ctx) -> Optional[Dict]:
    """Extract ~56 features from a single flow."""
    from scipy import stats as scipy_stats

    lengths = packets["Length"].values.astype(float)
    timestamps = packets["timestamp"]

    if len(lengths) < 3:
        return None

    time_diffs = timestamps.diff().dt.total_seconds().dropna().values * 1000
    if len(time_diffs) == 0:
        time_diffs = np.array([0.0])

    dur = max((timestamps.max() - timestamps.min()).total_seconds(), 0.001)
    f = {}

    # Basic stats
    f["flow_duration_sec"] = dur
    f["total_packets"] = len(lengths)
    f["total_bytes"] = float(lengths.sum())
    f["packets_per_sec"] = len(lengths) / dur
    f["bytes_per_sec"] = float(lengths.sum()) / dur

    # Packet size stats
    f["pkt_size_min"] = float(lengths.min())
    f["pkt_size_max"] = float(lengths.max())
    f["pkt_size_mean"] = float(lengths.mean())
    f["pkt_size_std"] = float(lengths.std()) if len(lengths) > 1 else 0.0
    f["pkt_size_median"] = float(np.median(lengths))
    f["pkt_size_q25"] = float(np.percentile(lengths, 25))
    f["pkt_size_q75"] = float(np.percentile(lengths, 75))
    f["pkt_size_iqr"] = f["pkt_size_q75"] - f["pkt_size_q25"]

    if len(lengths) > 2 and lengths.std() > 0:
        f["pkt_size_skewness"] = float(scipy_stats.skew(lengths))
        f["pkt_size_kurtosis"] = float(scipy_stats.kurtosis(lengths))
    else:
        f["pkt_size_skewness"] = 0.0
        f["pkt_size_kurtosis"] = 0.0

    # IAT stats
    f["iat_min_ms"] = float(time_diffs.min())
    f["iat_max_ms"] = float(time_diffs.max())
    f["iat_mean_ms"] = float(time_diffs.mean())
    f["iat_std_ms"] = float(time_diffs.std()) if len(time_diffs) > 1 else 0.0
    f["iat_median_ms"] = float(np.median(time_diffs))
    f["iat_q25_ms"] = float(np.percentile(time_diffs, 25)) if len(time_diffs) >= 4 else f["iat_mean_ms"]
    f["iat_q75_ms"] = float(np.percentile(time_diffs, 75)) if len(time_diffs) >= 4 else f["iat_mean_ms"]

    # Protocol encoding
    proto_map = {"TCP": 0, "UDP": 1, "QUIC": 2, "TLS": 3, "TLSv1.2": 3,
                 "TLSv1.3": 4, "HTTP": 5, "HTTPS": 6, "DNS": 7, "SSL": 3,
                 "STUN": 8, "DTLS": 9, "ICMPv6": 10, "ICMP": 11}
    f["protocol_encoded"] = proto_map.get(protocol, 12)
    f["is_encrypted"] = 1.0 if protocol in ("QUIC", "TLS", "TLSv1.2", "TLSv1.3", "SSL", "HTTPS", "DTLS") else 0.0

    # NOVELTY 1: Entropy
    f["pkt_size_entropy"] = compute_entropy(lengths)
    pos_iats = time_diffs[time_diffs > 0]
    f["iat_entropy"] = compute_entropy(pos_iats) if len(pos_iats) > 0 else 0.0

    # NOVELTY 2: Burst detection
    f.update(detect_bursts(time_diffs, lengths))

    # NOVELTY 3: First-N packet sizes (handshake fingerprint)
    for i in range(FIRST_N_PACKETS):
        f[f"first_pkt_{i}"] = float(lengths[i]) if i < len(lengths) else 0.0

    # NOVELTY 4: Context-aware host features
    c = ctx.get(src, {})
    f["ctx_dst_ip_count"] = c.get("src_dst_ip_count", 0)
    f["ctx_protocol_diversity"] = c.get("src_protocol_diversity", 0)
    f["ctx_connection_rate"] = c.get("src_connection_rate", 0.0)
    f["ctx_packets_per_sec"] = c.get("src_packets_per_sec", 0.0)
    f["ctx_bytes_per_sec"] = c.get("src_bytes_per_sec", 0.0)

    # Derived ratios
    f["avg_payload_ratio"] = f["pkt_size_mean"] / max(f["pkt_size_max"], 1)
    f["iat_coefficient_of_variation"] = f["iat_std_ms"] / max(f["iat_mean_ms"], 0.001)
    f["pkt_size_coefficient_of_variation"] = f["pkt_size_std"] / max(f["pkt_size_mean"], 0.001)

    # Labels
    f["label"] = label
    f["app_name"] = app_name
    f["protocol"] = protocol

    return f


def parse_csv(path: str) -> pd.DataFrame:
    """Parse Kaggle 5G CSV with encoding fallback."""
    for enc in ["utf-8", "latin-1", "euc-kr", "cp949"]:
        try:
            df = pd.read_csv(path, dtype=str, on_bad_lines="skip", encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        return pd.DataFrame()

    df.columns = [c.strip().strip('"') for c in df.columns]
    required = {"No.", "Time", "Source", "Destination", "Protocol", "Length"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df["No."] = pd.to_numeric(df["No."].str.strip('"'), errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"].str.strip('"'), errors="coerce")
    df["Source"] = df["Source"].str.strip('"')
    df["Destination"] = df["Destination"].str.strip('"')
    df["Protocol"] = df["Protocol"].str.strip('"')
    df["Time"] = df["Time"].str.strip('"')
    df["timestamp"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["No.", "Length", "timestamp"])
    return df


# ─────────────────────────────────────────────────────────────
# Main Processing Loop
# ─────────────────────────────────────────────────────────────

print("╔══════════════════════════════════════════════════════════╗")
print("║  FlowEmbed — 5G Traffic → Flow Feature Extraction       ║")
print("║  Processing on Kaggle Cloud                             ║")
print("╚══════════════════════════════════════════════════════════╝\n")

all_flows = []
total_packets = 0
total_flows = 0

for category_dir in sorted(INPUT_DIR.iterdir()):
    if not category_dir.is_dir():
        continue

    label = CATEGORY_LABEL_MAP.get(category_dir.name)
    if label is None:
        continue

    print(f"\n{'='*60}")
    print(f"Category: {category_dir.name} → label: {label}")
    print(f"{'='*60}")

    for app_dir in sorted(category_dir.iterdir()):
        if not app_dir.is_dir():
            continue

        app_name = app_dir.name
        csv_files = sorted(app_dir.glob("*.csv"))[:MAX_FILES_PER_APP]
        print(f"  App: {app_name} ({len(csv_files)} files)")

        for csv_file in tqdm(csv_files, desc=f"    {app_name}", leave=False):
            try:
                packets = parse_csv(str(csv_file))
                if packets.empty:
                    continue

                if len(packets) > MAX_PACKETS_PER_FILE:
                    packets = packets.head(MAX_PACKETS_PER_FILE)

                total_packets += len(packets)
                packets = packets.sort_values("timestamp").reset_index(drop=True)

                # Context-aware features
                src_ctx = compute_source_ip_context(packets)

                # Aggregate to flows
                flows_data = []
                for (src, dst, proto), group in packets.groupby(["Source", "Destination", "Protocol"]):
                    if len(group) < 3:
                        continue
                    feat = extract_flow_features(group, src, dst, proto, label, app_name, src_ctx)
                    if feat:
                        flows_data.append(feat)

                if flows_data:
                    all_flows.extend(flows_data)
                    total_flows += len(flows_data)
                    print(f"    {csv_file.name}: {len(packets):,} pkts → {len(flows_data)} flows")

            except Exception as e:
                print(f"    ERROR {csv_file.name}: {e}")
                continue

# ─────────────────────────────────────────────────────────────
# Save Output
# ─────────────────────────────────────────────────────────────
if all_flows:
    combined = pd.DataFrame(all_flows)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"  Total packets processed: {total_packets:,}")
    print(f"  Total flows extracted:   {total_flows:,}")
    print(f"  Features per flow:       {len(combined.columns) - 3}")  # minus label, app, protocol
    print(f"  Output file:             {OUTPUT_CSV}")
    print(f"  File size:               {OUTPUT_CSV.stat().st_size / (1024*1024):.1f} MB")
    print(f"\n  Class distribution:")
    for lbl, cnt in combined["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,} flows")
    print(f"\n  App distribution:")
    for app, cnt in combined["app_name"].value_counts().items():
        print(f"    {app}: {cnt:,} flows")
    print(f"{'='*60}")
    print(f"\n📥 Download: {OUTPUT_CSV}")
    print(f"   Copy this file to your local: Samsung/data/processed/kaggle_5g_flows.csv")
else:
    print("❌ No flows extracted!")
