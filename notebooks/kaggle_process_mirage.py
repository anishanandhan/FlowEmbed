"""
Kaggle Notebook Script — Process MIRAGE-2019 Mobile Traffic Dataset
===================================================================

INSTRUCTIONS:
1. Open your Kaggle notebook with the MIRAGE-2019 dataset loaded.
2. Copy-paste this script into a cell and run it.
3. Download the output CSV from /kaggle/working/kaggle_mirage_flows.csv
4. Place it in your local Samsung/data/processed/ folder.
"""

import os
import json
import glob
import math
import numpy as np
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INPUT_DIR = "/kaggle/input"
OUTPUT_CSV = "/kaggle/working/kaggle_mirage_flows.csv"

APP_LABEL_MAP = {
    "youtube": "streaming", "netflix": "streaming", "spotify": "streaming",
    "deezer": "streaming", "soundcloud": "streaming", "dailymotion": "streaming",
    "vimeo": "streaming", "tunein": "streaming", "pandora": "streaming",
    "shazam": "streaming", "twitch": "streaming",
    "whatsapp": "voip", "skype": "voip", "telegram": "voip",
    "hangouts": "voip", "viber": "voip", "signal": "voip",
    "line": "voip", "wechat": "voip", "discord": "voip",
    "messenger": "voip", "facetime": "voip", "zoom": "voip",
    "slither": "gaming", "pokemon": "gaming", "candy": "gaming",
    "clash": "gaming", "pubg": "gaming", "fortnite": "gaming",
    "minecraft": "gaming", "roblox": "gaming", "brawl": "gaming",
    "supercell": "gaming", "gameloft": "gaming", "rovio": "gaming",
    "king.com": "gaming", "zynga": "gaming", "game": "gaming",
    "instagram": "streaming", "facebook": "streaming",
    "twitter": "streaming", "tiktok": "streaming",
    "snapchat": "streaming", "pinterest": "streaming",
    "tumblr": "streaming", "reddit": "streaming",
}


def get_label_from_filename(filename):
    name = filename.lower()
    for keyword, label in APP_LABEL_MAP.items():
        if keyword in name:
            return label
    return None


def safe_float(val, default=0.0):
    """Safely convert a value to float, handling NaN and None."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except (ValueError, TypeError):
        return default


def detect_bursts(iats_ms, lengths, threshold_ms=50.0):
    empty = {
        "burst_count": 0, "burst_mean_size": 0.0, "burst_mean_duration_ms": 0.0,
        "burst_rate": 0.0, "inter_burst_mean_ms": 0.0, "burst_bytes_ratio": 0.0,
    }
    if len(iats_ms) < 2:
        return empty
    bursts, start, pkt_count = [], 0, 1
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
    total_bytes = float(lengths.sum()) if lengths.sum() > 0 else 1.0
    total_dur = float(iats_ms.sum()) if iats_ms.sum() > 0 else 1.0
    burst_bytes = sum(b["bytes"] for b in bursts)
    return {
        "burst_count": len(bursts),
        "burst_mean_size": np.mean([b["packets"] for b in bursts]),
        "burst_mean_duration_ms": np.mean([b["duration_ms"] for b in bursts]),
        "burst_rate": len(bursts) / (total_dur / 1000.0) if total_dur > 0 else 0.0,
        "inter_burst_mean_ms": total_dur / max(len(bursts) - 1, 1),
        "burst_bytes_ratio": burst_bytes / total_bytes if total_bytes > 0 else 0.0,
    }


def compute_entropy(values, num_bins=20):
    if len(values) < 2:
        return 0.0
    try:
        hist, _ = np.histogram(values, bins=num_bins, density=False)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    except:
        return 0.0


def extract_flow_features(flow_key, flow_data):
    """Extract 60 features from Mirage JSON using the EXACT known structure."""

    pkt_data = flow_data.get("packet_data", {})
    flow_feat = flow_data.get("flow_features", {})
    meta = flow_data.get("flow_metadata", {})

    # ── Raw packet arrays from packet_data ──
    raw_sizes = np.array([abs(x) for x in pkt_data.get("L4_payload_bytes", [])], dtype=float)
    raw_iats = np.array([abs(x) for x in pkt_data.get("iat", [])], dtype=float)

    total_pkts = len(raw_sizes)
    if total_pkts == 0:
        return None

    total_bytes = float(raw_sizes.sum())

    # ── Pre-computed stats from flow_features ──
    pkt_len = flow_feat.get("packet_length", {})
    iat_feat = flow_feat.get("iat", {})

    pkt_bi = pkt_len.get("biflow", {})
    iat_bi = iat_feat.get("biflow", {})

    # Duration: sum of all IATs gives total flow duration
    dur_ms = float(raw_iats.sum()) if len(raw_iats) > 0 else 0.0
    # Also check metadata for duration
    if isinstance(meta, dict) and "duration" in meta:
        meta_dur = safe_float(meta.get("duration"))
        if meta_dur > dur_ms:
            dur_ms = meta_dur

    dur_sec = max(dur_ms / 1000.0, 0.001)

    features = {}

    # ── Basic flow stats ──
    features["flow_duration_sec"] = dur_sec
    features["total_packets"] = total_pkts
    features["total_bytes"] = total_bytes
    features["packets_per_sec"] = total_pkts / dur_sec
    features["bytes_per_sec"] = total_bytes / dur_sec

    # ── Packet size stats (from pre-computed biflow features) ──
    features["pkt_size_min"] = safe_float(pkt_bi.get("min"))
    features["pkt_size_max"] = safe_float(pkt_bi.get("max"))
    features["pkt_size_mean"] = safe_float(pkt_bi.get("mean"))
    features["pkt_size_std"] = safe_float(pkt_bi.get("std"))
    features["pkt_size_median"] = safe_float(pkt_bi.get("50_percentile"))
    features["pkt_size_q25"] = safe_float(pkt_bi.get("30_percentile"))  # Closest to Q25
    features["pkt_size_q75"] = safe_float(pkt_bi.get("70_percentile"))  # Closest to Q75
    features["pkt_size_iqr"] = features["pkt_size_q75"] - features["pkt_size_q25"]
    features["pkt_size_skewness"] = safe_float(pkt_bi.get("skew"))
    features["pkt_size_kurtosis"] = safe_float(pkt_bi.get("kurtosis"))

    # ── IAT stats (from pre-computed biflow features) ──
    features["iat_min_ms"] = safe_float(iat_bi.get("min"))
    features["iat_max_ms"] = safe_float(iat_bi.get("max"))
    features["iat_mean_ms"] = safe_float(iat_bi.get("mean"))
    features["iat_std_ms"] = safe_float(iat_bi.get("std"))
    features["iat_median_ms"] = safe_float(iat_bi.get("50_percentile"))
    features["iat_q25_ms"] = safe_float(iat_bi.get("30_percentile"))
    features["iat_q75_ms"] = safe_float(iat_bi.get("70_percentile"))

    # ── Protocol & encryption ──
    parts = flow_key.split(",")
    features["protocol_encoded"] = int(parts[-1]) if len(parts) >= 5 else 6
    dst_port = int(parts[3]) if len(parts) >= 4 else 0
    features["is_encrypted"] = 1 if dst_port == 443 else 0

    # ── Entropy ──
    features["pkt_size_entropy"] = compute_entropy(raw_sizes) if len(raw_sizes) > 1 else 0.0
    features["iat_entropy"] = compute_entropy(raw_iats) if len(raw_iats) > 1 else 0.0

    # ── Burst detection ──
    burst = detect_bursts(raw_iats, raw_sizes)
    features.update(burst)

    # ── First 20 packet sizes (handshake fingerprint) ──
    for i in range(20):
        features[f"first_pkt_{i}"] = float(raw_sizes[i]) if i < len(raw_sizes) else 0.0

    # ── Context features ──
    features["ctx_dst_ip_count"] = 1
    features["ctx_protocol_diversity"] = 1
    features["ctx_connection_rate"] = total_pkts / dur_sec
    features["ctx_packets_per_sec"] = total_pkts / dur_sec
    features["ctx_bytes_per_sec"] = total_bytes / dur_sec

    # ── Ratio features ──
    features["avg_payload_ratio"] = features["pkt_size_mean"] / max(features["pkt_size_max"], 1)
    features["iat_coefficient_of_variation"] = features["iat_std_ms"] / max(features["iat_mean_ms"], 0.001)
    features["pkt_size_coefficient_of_variation"] = features["pkt_size_std"] / max(features["pkt_size_mean"], 0.001)

    return features


# ─────────────────────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────────────────────
def process_mirage():
    print(f"Searching for JSON biflow files in {INPUT_DIR}...")

    json_files = glob.glob(f"{INPUT_DIR}/**/*.json", recursive=True)
    print(f"Found {len(json_files)} JSON files total")

    if not json_files:
        print("No JSON files found!")
        return

    labeled_files = []
    unlabeled_apps = set()
    for f in json_files:
        label = get_label_from_filename(os.path.basename(f))
        if label:
            labeled_files.append((f, label))
        else:
            name = os.path.basename(f)
            unlabeled_apps.add(name.split("_MIRAGE")[0] if "_MIRAGE" in name else name[:40])

    print(f"\nMatched {len(labeled_files)} files to our target labels")
    if unlabeled_apps:
        print(f"Unmatched apps (skipped): {list(unlabeled_apps)[:10]}")

    label_counts = Counter(l for _, l in labeled_files)
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} files")

    all_flows = []
    errors = 0
    skipped_small = 0

    for i, (json_path, label) in enumerate(labeled_files):
        try:
            if i % 100 == 0:
                print(f"  Processing file {i+1}/{len(labeled_files)}... ({len(all_flows)} flows so far)")

            with open(json_path, "r") as f:
                data = json.load(f)

            app_name = os.path.basename(json_path)

            for flow_key, flow_data in data.items():
                if not isinstance(flow_data, dict):
                    continue
                if "packet_data" not in flow_data:
                    continue

                try:
                    features = extract_flow_features(flow_key, flow_data)

                    if features is None:
                        skipped_small += 1
                        continue

                    # Skip flows with fewer than 3 packets
                    if features["total_packets"] < 3:
                        skipped_small += 1
                        continue

                    features["label"] = label
                    features["app_name"] = app_name
                    all_flows.append(features)

                except Exception as e:
                    errors += 1
                    continue

        except Exception as e:
            errors += 1
            continue

    if not all_flows:
        print(f"\nNo flows extracted! Errors: {errors}, Skipped small: {skipped_small}")
        return

    df = pd.DataFrame(all_flows)

    # Replace any remaining NaN/inf with 0
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"✅ MIRAGE-2019 PROCESSING COMPLETE!")
    print(f"  Total flows extracted: {len(df):,}")
    print(f"  Features per flow:     {len(df.columns) - 2}")
    print(f"  Skipped (too small):   {skipped_small}")
    print(f"  Errors:                {errors}")
    print(f"\n  Class distribution:")
    for lbl, cnt in df["label"].value_counts().items():
        print(f"    {lbl}: {cnt:,} flows")
    print(f"{'='*60}")
    print(f"\n📥 Download: {OUTPUT_CSV}")
    print(f"   Copy to local: Samsung/data/processed/kaggle_mirage_flows.csv")


if __name__ == "__main__":
    process_mirage()
