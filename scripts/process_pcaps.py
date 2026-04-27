"""
Process PCAPs — Convert raw packet-level CSVs (Kaggle 5G) to flow-level features.

This script aggregates raw Wireshark-exported packet CSVs into flows using 
5-tuple grouping and extracts rich statistical features.

=== NOVELTY (inspired by FlowXpert + our own additions) ===

1. CONTEXT-AWARE HOST FEATURES (from FlowXpert):
   - Source IP fan-out: number of unique destination IPs contacted
   - Source port diversity: number of unique source ports used
   - Connection rate: new connections per second from source
   - These capture behavioral patterns of the HOST, not just the flow

2. PACKET SEQUENCE FEATURES (our novelty):
   - First-N packet sizes as a fixed-length sequence → fed to Transformer encoder
   - Captures the "handshake signature" that differs across protocols
   - Encrypted traffic (QUIC vs TLS) has distinctive early packet patterns

3. BURST DETECTION FEATURES (our novelty):
   - Burst count, mean burst size, burst rate
   - Streaming traffic bursts differently than gaming or VoIP
   - Detects the "rhythm" of traffic — a unique behavioral fingerprint

4. ENTROPY-BASED FEATURES (our novelty):
   - Packet size entropy, inter-arrival time entropy
   - High entropy = unpredictable (browsing), low entropy = regular (streaming)
   - Works on encrypted traffic without decryption

Usage:
    python scripts/process_pcaps.py
    python scripts/process_pcaps.py --dataset kaggle_5g --output data/processed/kaggle_5g_flows.csv
    python scripts/process_pcaps.py --dataset samsung_s23
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FIRST_N_PACKETS = 20      # Number of first packets to capture as sequence
FLOW_TIMEOUT_SEC = 120    # Max idle time before a flow expires
BURST_THRESHOLD_MS = 50   # Max inter-arrival time within a burst


# ─────────────────────────────────────────────────────────────
# Core: Packet → Flow aggregation
# ─────────────────────────────────────────────────────────────

def parse_kaggle_5g_csv(csv_path: str) -> pd.DataFrame:
    """
    Parse a Kaggle 5G traffic dataset CSV (Wireshark export format).

    Columns: No., Time, Source, Destination, Protocol, Length, Info
    """
    # Try multiple encodings (Korean dataset may use EUC-KR)
    for encoding in ["utf-8", "latin-1", "euc-kr", "cp949"]:
        try:
            df = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip", encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        logger.warning(f"Failed to read {csv_path} with any encoding")
        return pd.DataFrame()

    # Standardize column names
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Required columns
    required = {"No.", "Time", "Source", "Destination", "Protocol", "Length"}
    if not required.issubset(set(df.columns)):
        logger.warning(f"Missing columns in {csv_path}. Found: {df.columns.tolist()}")
        return pd.DataFrame()

    # Clean and convert types
    df["No."] = pd.to_numeric(df["No."].str.strip('"'), errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"].str.strip('"'), errors="coerce")
    df["Source"] = df["Source"].str.strip('"')
    df["Destination"] = df["Destination"].str.strip('"')
    df["Protocol"] = df["Protocol"].str.strip('"')

    # Parse timestamps
    df["Time"] = df["Time"].str.strip('"')
    df["timestamp"] = pd.to_datetime(df["Time"], errors="coerce")

    # Drop rows with parsing failures
    df = df.dropna(subset=["No.", "Length", "timestamp"])

    return df


def aggregate_packets_to_flows(
    packets: pd.DataFrame,
    label: str,
    app_name: str,
) -> pd.DataFrame:
    """
    Aggregate raw packets into flows using (Source, Destination, Protocol) as key.
    Extract comprehensive flow-level features.
    """
    if packets.empty:
        return pd.DataFrame()

    packets = packets.sort_values("timestamp").reset_index(drop=True)

    # Group by (src_ip, dst_ip, protocol) — simplified 3-tuple since ports
    # aren't always available in the Kaggle dataset
    flows_data = []

    grouped = packets.groupby(["Source", "Destination", "Protocol"])

    # Collect all source IPs for context-aware features
    all_source_ips = packets["Source"].unique()

    # Pre-compute source IP context (FlowXpert-inspired)
    src_ip_context = compute_source_ip_context(packets)

    for (src, dst, proto), group in grouped:
        if len(group) < 3:  # Skip very short flows (noise)
            continue

        flow_features = extract_flow_features(
            group, src, dst, proto, label, app_name, src_ip_context
        )

        if flow_features is not None:
            flows_data.append(flow_features)

    if not flows_data:
        return pd.DataFrame()

    return pd.DataFrame(flows_data)


def compute_source_ip_context(packets: pd.DataFrame) -> Dict:
    """
    Compute context-aware features for each source IP.

    === FlowXpert-inspired novelty ===
    These features capture the BEHAVIORAL PATTERN of the source host,
    not just individual flows. A streaming client behaves differently from
    a port scanner at the host level.
    """
    context = {}

    for src_ip, group in packets.groupby("Source"):
        timestamps = group["timestamp"]
        duration = (timestamps.max() - timestamps.min()).total_seconds()
        duration = max(duration, 0.001)  # Avoid division by zero

        context[src_ip] = {
            # Fan-out: how many unique destinations does this source contact?
            "src_dst_ip_count": group["Destination"].nunique(),
            # Port diversity (using protocol as proxy since ports aren't in data)
            "src_protocol_diversity": group["Protocol"].nunique(),
            # Connection rate: unique flows per second
            "src_flow_count": len(group.groupby(["Destination", "Protocol"])),
            "src_connection_rate": len(group.groupby(["Destination", "Protocol"])) / duration,
            # Packet rate from this source
            "src_total_packets": len(group),
            "src_packets_per_sec": len(group) / duration,
            # Byte rate from this source
            "src_total_bytes": group["Length"].sum(),
            "src_bytes_per_sec": group["Length"].sum() / duration,
        }

    return context


def extract_flow_features(
    packets: pd.DataFrame,
    src: str,
    dst: str,
    protocol: str,
    label: str,
    app_name: str,
    src_ip_context: Dict,
) -> Optional[Dict]:
    """
    Extract comprehensive features from a single flow's packets.

    Returns a dict with ~50+ features per flow.
    """
    lengths = packets["Length"].values.astype(float)
    timestamps = packets["timestamp"]

    if len(lengths) < 3:
        return None

    # Compute inter-arrival times (IAT) in milliseconds
    time_diffs = timestamps.diff().dt.total_seconds().dropna().values * 1000
    if len(time_diffs) == 0:
        time_diffs = np.array([0.0])

    flow_duration = (timestamps.max() - timestamps.min()).total_seconds()
    flow_duration = max(flow_duration, 0.001)

    features = {}

    # ─────────────────────────────────────────────────
    # 1. Basic flow statistics
    # ─────────────────────────────────────────────────
    features["flow_duration_sec"] = flow_duration
    features["total_packets"] = len(lengths)
    features["total_bytes"] = float(lengths.sum())
    features["packets_per_sec"] = len(lengths) / flow_duration
    features["bytes_per_sec"] = float(lengths.sum()) / flow_duration

    # ─────────────────────────────────────────────────
    # 2. Packet size statistics
    # ─────────────────────────────────────────────────
    features["pkt_size_min"] = float(lengths.min())
    features["pkt_size_max"] = float(lengths.max())
    features["pkt_size_mean"] = float(lengths.mean())
    features["pkt_size_std"] = float(lengths.std()) if len(lengths) > 1 else 0.0
    features["pkt_size_median"] = float(np.median(lengths))

    # Quartiles
    features["pkt_size_q25"] = float(np.percentile(lengths, 25))
    features["pkt_size_q75"] = float(np.percentile(lengths, 75))
    features["pkt_size_iqr"] = features["pkt_size_q75"] - features["pkt_size_q25"]

    # Skewness and kurtosis
    if len(lengths) > 2 and lengths.std() > 0:
        from scipy import stats as scipy_stats
        features["pkt_size_skewness"] = float(scipy_stats.skew(lengths))
        features["pkt_size_kurtosis"] = float(scipy_stats.kurtosis(lengths))
    else:
        features["pkt_size_skewness"] = 0.0
        features["pkt_size_kurtosis"] = 0.0

    # ─────────────────────────────────────────────────
    # 3. Inter-arrival time statistics
    # ─────────────────────────────────────────────────
    features["iat_min_ms"] = float(time_diffs.min()) if len(time_diffs) > 0 else 0.0
    features["iat_max_ms"] = float(time_diffs.max()) if len(time_diffs) > 0 else 0.0
    features["iat_mean_ms"] = float(time_diffs.mean()) if len(time_diffs) > 0 else 0.0
    features["iat_std_ms"] = float(time_diffs.std()) if len(time_diffs) > 1 else 0.0
    features["iat_median_ms"] = float(np.median(time_diffs)) if len(time_diffs) > 0 else 0.0

    # IAT quartiles
    if len(time_diffs) >= 4:
        features["iat_q25_ms"] = float(np.percentile(time_diffs, 25))
        features["iat_q75_ms"] = float(np.percentile(time_diffs, 75))
    else:
        features["iat_q25_ms"] = features["iat_mean_ms"]
        features["iat_q75_ms"] = features["iat_mean_ms"]

    # ─────────────────────────────────────────────────
    # 4. Protocol encoding
    # ─────────────────────────────────────────────────
    protocol_map = {
        "TCP": 0, "UDP": 1, "QUIC": 2, "TLS": 3, "TLSv1.2": 3,
        "TLSv1.3": 4, "HTTP": 5, "HTTPS": 6, "DNS": 7,
        "SSL": 3, "STUN": 8, "DTLS": 9, "ICMPv6": 10, "ICMP": 11,
    }
    features["protocol_encoded"] = protocol_map.get(protocol, 12)
    features["is_encrypted"] = 1.0 if protocol in (
        "QUIC", "TLS", "TLSv1.2", "TLSv1.3", "SSL", "HTTPS", "DTLS"
    ) else 0.0

    # ─────────────────────────────────────────────────
    # 5. NOVELTY: Packet size entropy
    #    High entropy = unpredictable sizes (browsing)
    #    Low entropy = regular sizes (streaming, VoIP)
    # ─────────────────────────────────────────────────
    features["pkt_size_entropy"] = compute_entropy(lengths)
    features["iat_entropy"] = compute_entropy(time_diffs[time_diffs > 0]) if len(time_diffs[time_diffs > 0]) > 0 else 0.0

    # ─────────────────────────────────────────────────
    # 6. NOVELTY: Burst detection features
    #    Streaming sends data in bursts; gaming is more uniform
    # ─────────────────────────────────────────────────
    burst_features = detect_bursts(time_diffs, lengths)
    features.update(burst_features)

    # ─────────────────────────────────────────────────
    # 7. NOVELTY: First-N packet sizes (sequence feature)
    #    Captures the "protocol handshake fingerprint"
    #    QUIC vs TLS have very different initial sequences
    # ─────────────────────────────────────────────────
    for i in range(FIRST_N_PACKETS):
        if i < len(lengths):
            features[f"first_pkt_{i}"] = float(lengths[i])
        else:
            features[f"first_pkt_{i}"] = 0.0

    # ─────────────────────────────────────────────────
    # 8. NOVELTY: Context-aware host features (FlowXpert)
    # ─────────────────────────────────────────────────
    ctx = src_ip_context.get(src, {})
    features["ctx_dst_ip_count"] = ctx.get("src_dst_ip_count", 0)
    features["ctx_protocol_diversity"] = ctx.get("src_protocol_diversity", 0)
    features["ctx_connection_rate"] = ctx.get("src_connection_rate", 0.0)
    features["ctx_packets_per_sec"] = ctx.get("src_packets_per_sec", 0.0)
    features["ctx_bytes_per_sec"] = ctx.get("src_bytes_per_sec", 0.0)

    # ─────────────────────────────────────────────────
    # 9. Derived ratios
    # ─────────────────────────────────────────────────
    features["avg_payload_ratio"] = features["pkt_size_mean"] / max(features["pkt_size_max"], 1)
    features["iat_coefficient_of_variation"] = (
        features["iat_std_ms"] / max(features["iat_mean_ms"], 0.001)
    )
    features["pkt_size_coefficient_of_variation"] = (
        features["pkt_size_std"] / max(features["pkt_size_mean"], 0.001)
    )

    # ─────────────────────────────────────────────────
    # Labels
    # ─────────────────────────────────────────────────
    features["label"] = label
    features["app_name"] = app_name
    features["protocol"] = protocol

    return features


def compute_entropy(values: np.ndarray, num_bins: int = 20) -> float:
    """
    Compute Shannon entropy of a distribution.
    Higher entropy = more unpredictable = more diverse packet patterns.
    """
    if len(values) < 2:
        return 0.0

    # Bin the values and compute histogram probabilities
    try:
        hist, _ = np.histogram(values, bins=num_bins, density=False)
        hist = hist[hist > 0]  # Remove zero bins
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return float(entropy)
    except Exception:
        return 0.0


def detect_bursts(
    iats_ms: np.ndarray,
    lengths: np.ndarray,
    threshold_ms: float = BURST_THRESHOLD_MS,
) -> Dict:
    """
    Detect burst patterns in traffic.

    A burst = consecutive packets with IAT < threshold.
    Streaming: long bursts (video chunks), then pause
    Gaming: short, regular bursts (game state updates)
    VoIP: very regular, uniform bursts (codec frames)
    """
    if len(iats_ms) < 2:
        return {
            "burst_count": 0,
            "burst_mean_size": 0.0,
            "burst_mean_duration_ms": 0.0,
            "burst_rate": 0.0,
            "inter_burst_mean_ms": 0.0,
            "burst_bytes_ratio": 0.0,
        }

    bursts = []
    current_burst_start = 0
    current_burst_packets = 1

    for i, iat in enumerate(iats_ms):
        if iat < threshold_ms:
            current_burst_packets += 1
        else:
            if current_burst_packets >= 2:
                burst_lengths = lengths[current_burst_start:current_burst_start + current_burst_packets]
                burst_iats = iats_ms[current_burst_start:current_burst_start + current_burst_packets - 1]
                bursts.append({
                    "packets": current_burst_packets,
                    "bytes": float(burst_lengths.sum()),
                    "duration_ms": float(burst_iats.sum()) if len(burst_iats) > 0 else 0.0,
                })
            current_burst_start = i + 1
            current_burst_packets = 1

    # Handle last burst
    if current_burst_packets >= 2:
        burst_lengths = lengths[current_burst_start:current_burst_start + current_burst_packets]
        burst_iats = iats_ms[current_burst_start:current_burst_start + current_burst_packets - 1]
        bursts.append({
            "packets": current_burst_packets,
            "bytes": float(burst_lengths.sum()),
            "duration_ms": float(burst_iats.sum()) if len(burst_iats) > 0 else 0.0,
        })

    total_bytes = float(lengths.sum())
    total_duration = float(iats_ms.sum()) if len(iats_ms) > 0 else 1.0

    if bursts:
        burst_bytes = sum(b["bytes"] for b in bursts)
        burst_durations = [b["duration_ms"] for b in bursts]

        return {
            "burst_count": len(bursts),
            "burst_mean_size": np.mean([b["packets"] for b in bursts]),
            "burst_mean_duration_ms": np.mean(burst_durations),
            "burst_rate": len(bursts) / max(total_duration / 1000, 0.001),
            "inter_burst_mean_ms": (total_duration - sum(burst_durations)) / max(len(bursts), 1),
            "burst_bytes_ratio": burst_bytes / max(total_bytes, 1),
        }
    else:
        return {
            "burst_count": 0,
            "burst_mean_size": 0.0,
            "burst_mean_duration_ms": 0.0,
            "burst_rate": 0.0,
            "inter_burst_mean_ms": 0.0,
            "burst_bytes_ratio": 0.0,
        }


# ─────────────────────────────────────────────────────────────
# Dataset-specific processors
# ─────────────────────────────────────────────────────────────

def process_kaggle_5g_dataset(
    data_dir: str,
    output_path: str,
    max_files_per_app: int = 5,
    max_packets_per_file: int = 500000,
) -> pd.DataFrame:
    """
    Process the entire Kaggle 5G Traffic Dataset.

    Directory structure:
        5G_Traffic_Datasets/
        ├── Game_Streaming/      → gaming
        ├── Online_Game/         → gaming
        ├── Stored_Streaming/    → streaming
        ├── Live_Streaming/      → streaming
        ├── Video_Conferencing/  → voip
        └── Metaverse/           → xr_ar
    """
    data_dir = Path(data_dir)
    base_dir = data_dir / "5G_Traffic_Datasets"

    if not base_dir.exists():
        # Try without subdirectory
        base_dir = data_dir
        if not base_dir.exists():
            logger.error(f"Dataset directory not found: {data_dir}")
            return pd.DataFrame()

    # Category → label mapping
    category_label_map = {
        "Game_Streaming": "gaming",
        "Online_Game": "gaming",
        "Stored_Streaming": "streaming",
        "Live_Streaming": "streaming",
        "Video_Conferencing": "voip",
        "Metaverse": "xr_ar",
    }

    all_flows = []
    total_packets = 0
    total_flows = 0

    for category_dir in sorted(base_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name
        label = category_label_map.get(category_name)

        if label is None:
            logger.info(f"Skipping unknown category: {category_name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category_name} → label: {label}")
        logger.info(f"{'='*60}")

        # Each category has app subdirectories
        for app_dir in sorted(category_dir.iterdir()):
            if not app_dir.is_dir():
                continue

            app_name = app_dir.name
            csv_files = sorted(app_dir.glob("*.csv"))[:max_files_per_app]

            logger.info(f"  App: {app_name} ({len(csv_files)} files)")

            for csv_file in tqdm(csv_files, desc=f"    {app_name}", leave=False):
                try:
                    packets = parse_kaggle_5g_csv(str(csv_file))

                    if packets.empty:
                        continue

                    # Limit packets per file to manage memory
                    if len(packets) > max_packets_per_file:
                        packets = packets.head(max_packets_per_file)

                    total_packets += len(packets)

                    flows = aggregate_packets_to_flows(packets, label, app_name)

                    if not flows.empty:
                        all_flows.append(flows)
                        total_flows += len(flows)
                        logger.info(
                            f"    {csv_file.name}: {len(packets)} packets → {len(flows)} flows"
                        )

                except Exception as e:
                    logger.error(f"    Error processing {csv_file.name}: {e}")
                    continue

    if not all_flows:
        logger.error("No flows extracted!")
        return pd.DataFrame()

    combined = pd.concat(all_flows, ignore_index=True)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Processing complete!")
    logger.info(f"  Total packets processed: {total_packets:,}")
    logger.info(f"  Total flows extracted: {total_flows:,}")
    logger.info(f"  Output saved to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    logger.info(f"\n  Class distribution:")
    for label, count in combined["label"].value_counts().items():
        logger.info(f"    {label}: {count:,} flows")
    logger.info(f"{'='*60}")

    return combined


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Process raw packet data to flow features")
    parser.add_argument(
        "--dataset", type=str, default="kaggle_5g",
        choices=["kaggle_5g", "samsung_s23", "all"],
        help="Which dataset to process"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/processed/<dataset>_flows.csv)"
    )
    parser.add_argument(
        "--max-files", type=int, default=5,
        help="Max CSV files to process per app (for speed during development)"
    )
    parser.add_argument(
        "--max-packets", type=int, default=500000,
        help="Max packets per file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset in ("kaggle_5g", "all"):
        output = args.output or str(PROCESSED_DIR / "kaggle_5g_flows.csv")
        process_kaggle_5g_dataset(
            data_dir=str(RAW_DIR / "kaggle_5g"),
            output_path=output,
            max_files_per_app=args.max_files,
            max_packets_per_file=args.max_packets,
        )

    if args.dataset in ("samsung_s23", "all"):
        output = args.output or str(PROCESSED_DIR / "samsung_flows.csv")
        samsung_dir = RAW_DIR / "samsung_s23"
        if samsung_dir.exists() and any(samsung_dir.glob("*.pcap*")):
            from src.data.pcap_processor import process_samsung_captures
            process_samsung_captures(
                pcap_dir=str(samsung_dir),
                output_path=output,
            )
        else:
            logger.warning(f"No Samsung S23 PCAPs found in {samsung_dir}")
