"""
Process CESNET-QUIC22 raw data into FlowEmbed's 60-feature format.
Optimized version: processes in chunks, uses multiprocessing, and limits per-file extraction to be fast.
"""

import sys
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
RAW_DIR = Path("/Users/anishanan/Samsung/data/raw/cesnet_quic22/cesnet-quic22")
OUTPUT_CSV = Path("/Users/anishanan/Samsung/data/processed/cesnet_flows_processed.csv")

CATEGORY_MAP = {
    "Games": "gaming",
    "Streaming media": "streaming",
    "Music": "streaming",
    "Videoconferencing": "voip",
}

APP_MAP = {
    "youtube": "streaming",
    "spotify": "streaming",
    "google-hangouts": "voip",
    "easybrain": "gaming",
    "csgo-market": "gaming",
    "chess-com": "gaming",
    "blitz-gg": "gaming",
    "unitygames": "gaming",
    "gamedock": "gaming",
    "playradio": "streaming",
    "xhamster": "streaming",
}

BURST_THRESHOLD_MS = 50
FIRST_N_PACKETS = 20

# Limits
MAX_FLOWS_PER_FILE = 5000
CHUNK_SIZE = 50000


def compute_entropy(values, num_bins=20):
    if len(values) < 2:
        return 0.0
    try:
        hist, _ = np.histogram(values, bins=num_bins, density=False)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))
    except Exception:
        return 0.0


def detect_bursts(iats_ms, lengths, threshold_ms=BURST_THRESHOLD_MS):
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
                bursts.append({
                    "packets": pkt_count,
                    "bytes": float(bl.sum()),
                    "duration_ms": float(bi.sum()) if len(bi) > 0 else 0.0,
                })
            start, pkt_count = i + 1, 1

    if pkt_count >= 2:
        bl = lengths[start:start + pkt_count]
        bi = iats_ms[start:start + pkt_count - 1]
        bursts.append({
            "packets": pkt_count,
            "bytes": float(bl.sum()),
            "duration_ms": float(bi.sum()) if len(bi) > 0 else 0.0,
        })

    if not bursts:
        return empty

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


def process_row(row_tuple):
    # row_tuple elements corresponding to the df columns. We access by index or getattr.
    # We expect dict-like access for simplicity, so let's assume it's passed as a dict
    row = row_tuple
    app = str(row.get("APP", "")).lower()
    cat = str(row.get("CATEGORY", ""))
    
    label = None
    if app in APP_MAP:
        label = APP_MAP[app]
    elif cat in CATEGORY_MAP:
        label = CATEGORY_MAP[cat]
        
    if not label:
        return None

    try:
        ppi_raw = row.get("PPI", "[]")
        if isinstance(ppi_raw, str):
            ppi = json.loads(ppi_raw.replace("'", '"')) # json.loads is faster than ast.literal_eval
        else:
            ppi = ppi_raw

        if not ppi or len(ppi) < 3:
            return None

        timing_ms = np.array(ppi[0], dtype=float)
        sizes = np.abs(np.array(ppi[2], dtype=float))

        if len(sizes) < 3:
            return None

        total_packets = int(row.get("PACKETS", 0)) + int(row.get("PACKETS_REV", 0))
        total_bytes = float(row.get("BYTES", 0)) + float(row.get("BYTES_REV", 0))
        dur_sec = max(float(row.get("DURATION", 0.001)), 0.001)

        iats = timing_ms[1:] if len(timing_ms) > 1 else np.array([0.0])
        iats = np.abs(iats)

        f = {}
        f["flow_duration_sec"] = dur_sec
        f["total_packets"] = total_packets
        f["total_bytes"] = total_bytes
        f["packets_per_sec"] = total_packets / dur_sec
        f["bytes_per_sec"] = total_bytes / dur_sec

        f["pkt_size_min"] = float(sizes.min())
        f["pkt_size_max"] = float(sizes.max())
        f["pkt_size_mean"] = float(sizes.mean())
        f["pkt_size_std"] = float(sizes.std()) if len(sizes) > 1 else 0.0
        f["pkt_size_median"] = float(np.median(sizes))
        f["pkt_size_q25"] = float(np.percentile(sizes, 25))
        f["pkt_size_q75"] = float(np.percentile(sizes, 75))
        f["pkt_size_iqr"] = f["pkt_size_q75"] - f["pkt_size_q25"]

        if len(sizes) > 2 and sizes.std() > 0:
            f["pkt_size_skewness"] = float(scipy_stats.skew(sizes))
            f["pkt_size_kurtosis"] = float(scipy_stats.kurtosis(sizes))
        else:
            f["pkt_size_skewness"] = 0.0
            f["pkt_size_kurtosis"] = 0.0

        f["iat_min_ms"] = float(iats.min()) if len(iats) > 0 else 0.0
        f["iat_max_ms"] = float(iats.max()) if len(iats) > 0 else 0.0
        f["iat_mean_ms"] = float(iats.mean()) if len(iats) > 0 else 0.0
        f["iat_std_ms"] = float(iats.std()) if len(iats) > 1 else 0.0
        f["iat_median_ms"] = float(np.median(iats)) if len(iats) > 0 else 0.0
        f["iat_q25_ms"] = float(np.percentile(iats, 25)) if len(iats) >= 4 else f["iat_mean_ms"]
        f["iat_q75_ms"] = float(np.percentile(iats, 75)) if len(iats) >= 4 else f["iat_mean_ms"]

        f["protocol_encoded"] = 2
        f["is_encrypted"] = 1.0

        f["pkt_size_entropy"] = compute_entropy(sizes)
        pos_iats = iats[iats > 0]
        f["iat_entropy"] = compute_entropy(pos_iats) if len(pos_iats) > 0 else 0.0

        f.update(detect_bursts(iats, sizes))

        for i in range(FIRST_N_PACKETS):
            f[f"first_pkt_{i}"] = float(sizes[i]) if i < len(sizes) else 0.0

        f["ctx_dst_ip_count"] = 1
        f["ctx_protocol_diversity"] = 1
        f["ctx_connection_rate"] = 0.0
        f["ctx_packets_per_sec"] = f["packets_per_sec"]
        f["ctx_bytes_per_sec"] = f["bytes_per_sec"]

        f["avg_payload_ratio"] = f["pkt_size_mean"] / max(f["pkt_size_max"], 1)
        f["iat_coefficient_of_variation"] = f["iat_std_ms"] / max(f["iat_mean_ms"], 0.001)
        f["pkt_size_coefficient_of_variation"] = f["pkt_size_std"] / max(f["pkt_size_mean"], 0.001)

        f["label"] = label
        return f

    except Exception:
        return None

def process_file(gz_file):
    extracted = []
    try:
        # We only read up to a few chunks to find enough labeled flows quickly
        for chunk in pd.read_csv(gz_file, compression="gzip", low_memory=False, chunksize=CHUNK_SIZE):
            # Pre-filter by category/app
            chunk['APP'] = chunk['APP'].astype(str).str.lower()
            mask = (chunk['CATEGORY'].isin(CATEGORY_MAP.keys())) | (chunk['APP'].isin(APP_MAP.keys()))
            filtered = chunk[mask]
            
            if len(filtered) == 0:
                continue
            
            # Process rows
            dicts = filtered.to_dict('records')
            for d in dicts:
                res = process_row(d)
                if res:
                    extracted.append(res)
                if len(extracted) >= MAX_FLOWS_PER_FILE:
                    break
                    
            if len(extracted) >= MAX_FLOWS_PER_FILE:
                break
    except Exception as e:
        print(f"Error reading {gz_file}: {e}")
        
    return extracted

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  FlowEmbed — CESNET-QUIC22 Feature Extraction (Local)   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    gz_files = sorted(RAW_DIR.rglob("flows-*.csv.gz"))
    print(f"Found {len(gz_files)} gzipped flow CSVs.\n")

    all_flows = []
    
    # Process files in parallel to speed things up
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_file, gf): gf for gf in gz_files}
        for future in as_completed(future_to_file):
            gf = future_to_file[future]
            try:
                result = future.result()
                all_flows.extend(result)
                print(f"✅ Processed {gf.name}: {len(result)} flows extracted.")
            except Exception as e:
                print(f"❌ Error processing {gf.name}: {e}")

    if all_flows:
        final_df = pd.DataFrame(all_flows)
        final_df.to_csv(OUTPUT_CSV, index=False)

        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"  Total flows extracted:  {len(final_df):,}")
        print(f"  Features per flow:      {len(final_df.columns) - 1}")
        print(f"  Output file:            {OUTPUT_CSV}")
        print(f"\n  Class distribution:")
        for lbl, cnt in final_df["label"].value_counts().items():
            print(f"    {lbl}: {cnt:,}")
        print(f"{'='*60}")
    else:
        print("\n❌ No flows extracted!")
