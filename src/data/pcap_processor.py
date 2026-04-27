"""
PCAP Processor — Convert raw PCAP files to flow-level feature CSVs.

Uses nfstream as the primary engine (Python-native, fast).
CICFlowMeter available as fallback for offline processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def process_pcap_nfstream(
    pcap_path: str,
    label: Optional[str] = None,
    statistical_analysis: bool = True,
) -> pd.DataFrame:
    """
    Process a PCAP file using nfstream and extract flow-level features.

    Args:
        pcap_path: Path to the PCAP/PCAPNG file.
        label: Optional traffic class label to add to all flows.
        statistical_analysis: Whether to include statistical features (recommended).

    Returns:
        DataFrame with one row per flow and all extracted features.
    """
    try:
        from nfstream import NFStreamer
    except ImportError:
        raise ImportError(
            "nfstream is required. Install with: pip install nfstream\n"
            "On macOS, you may also need: brew install libpcap"
        )

    logger.info(f"Processing PCAP: {pcap_path}")

    streamer = NFStreamer(
        source=str(pcap_path),
        statistical_analysis=statistical_analysis,
        idle_timeout=120,       # Flow idle timeout in seconds
        active_timeout=1800,    # Flow active timeout in seconds
    )

    # Convert to DataFrame
    flows = streamer.to_pandas()

    if flows.empty:
        logger.warning(f"No flows extracted from {pcap_path}")
        return pd.DataFrame()

    # Add label if provided
    if label is not None:
        flows["label"] = label

    # Add source file info
    flows["source_file"] = Path(pcap_path).name

    logger.info(f"Extracted {len(flows)} flows from {pcap_path}")
    return flows


def process_pcap_live(
    interface: str = "en0",
    duration: int = 60,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Capture live traffic from a network interface using nfstream.

    Args:
        interface: Network interface name (e.g., 'en0' for Mac WiFi).
        duration: Capture duration in seconds.
        label: Optional traffic class label.

    Returns:
        DataFrame with flow features.
    """
    try:
        from nfstream import NFStreamer
    except ImportError:
        raise ImportError("nfstream is required. Install with: pip install nfstream")

    logger.info(f"Capturing live traffic on {interface} for {duration}s...")

    streamer = NFStreamer(
        source=interface,
        statistical_analysis=True,
        idle_timeout=30,
        active_timeout=duration,
    )

    flows = streamer.to_pandas()

    if label is not None:
        flows["label"] = label

    logger.info(f"Captured {len(flows)} live flows")
    return flows


def process_pcap_directory(
    pcap_dir: str,
    label_map: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Process all PCAP files in a directory.

    Args:
        pcap_dir: Directory containing PCAP files.
        label_map: Dict mapping filename patterns to labels.
                   E.g., {"youtube": "streaming", "gaming": "gaming"}
        output_path: Optional path to save combined CSV.

    Returns:
        Combined DataFrame from all PCAPs.
    """
    pcap_dir = Path(pcap_dir)
    all_flows = []

    pcap_files = list(pcap_dir.glob("*.pcap")) + list(pcap_dir.glob("*.pcapng"))

    if not pcap_files:
        logger.warning(f"No PCAP files found in {pcap_dir}")
        return pd.DataFrame()

    for pcap_file in pcap_files:
        # Determine label from filename if label_map provided
        label = None
        if label_map:
            fname_lower = pcap_file.stem.lower()
            for pattern, lbl in label_map.items():
                if pattern.lower() in fname_lower:
                    label = lbl
                    break

        try:
            flows = process_pcap_nfstream(str(pcap_file), label=label)
            if not flows.empty:
                all_flows.append(flows)
        except Exception as e:
            logger.error(f"Failed to process {pcap_file}: {e}")
            continue

    if not all_flows:
        return pd.DataFrame()

    combined = pd.concat(all_flows, ignore_index=True)

    if output_path:
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined)} flows to {output_path}")

    return combined


def process_samsung_captures(
    pcap_dir: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Process Samsung S23 Ultra captured PCAPs with app-specific labels.

    Expected file naming convention:
        youtube_4k.pcap, instagram_reels.pcap, bgmi_gaming.pcap,
        whatsapp_call.pcap, google_maps.pcap, idle_background.pcap
    """
    label_map = {
        "youtube": "streaming",
        "netflix": "streaming",
        "instagram": "social_media",
        "tiktok": "social_media",
        "bgmi": "gaming",
        "gaming": "gaming",
        "pubg": "gaming",
        "whatsapp": "voip",
        "zoom": "voip",
        "discord": "voip",
        "maps": "browsing",
        "chrome": "browsing",
        "safari": "browsing",
        "idle": "browsing",
        "download": "file_transfer",
        "drive": "file_transfer",
        "vpn": "vpn",
    }

    return process_pcap_directory(pcap_dir, label_map, output_path)
