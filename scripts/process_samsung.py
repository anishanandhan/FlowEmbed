"""
Samsung Galaxy S23 Ultra - Traffic Integration Script.
Processes raw `.pcap` captures from the Samsung S23 hotspot and converts
them into the exact 60-feature schema required by the FlowEmbed model.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nfstream_plugin import FlowEmbedPlugin, extract_features_from_flow
from src.config import RAW_DIR, PROCESSED_DIR

# Attempt to load nfstream
try:
    from nfstream import NFStreamer
except ImportError:
    print("❌ NFStream is required for PCAP processing. Install with: pip install nfstream")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("SamsungIntegration")

LABEL_MAP = {
    "youtube": "streaming",
    "netflix": "streaming",
    "instagram": "social_media",
    "bgmi": "gaming",
    "gaming": "gaming",
    "pubg": "gaming",
    "whatsapp": "voip",
    "zoom": "voip",
}

def process_samsung_pcap(pcap_path: Path, label: str) -> pd.DataFrame:
    """Processes a single S23 PCAP into 60-feature FlowEmbed format."""
    logger.info(f"Processing S23 capture: {pcap_path.name} -> {label}")
    
    streamer = NFStreamer(
        source=str(pcap_path),
        udps=FlowEmbedPlugin(),
        statistical_analysis=False # We handle stats manually in the plugin
    )
    
    flow_records = []
    
    for flow in streamer:
        # Ignore tiny uninteresting flows
        if flow.bidirectional_packets < 5:
            continue
            
        features_np = extract_features_from_flow(flow)
        
        # We need column names matching `train.csv`
        # We will use dummy names for now since the MLP just takes the raw numpy array
        # But to be safe and match the schema:
        record = {f"feat_{i}": features_np[i] for i in range(60)}
        record["label"] = label
        record["source_device"] = "Samsung Galaxy S23 Ultra"
        flow_records.append(record)
        
    return pd.DataFrame(flow_records)


def main():
    samsung_dir = RAW_DIR / "samsung_s23"
    samsung_dir.mkdir(parents=True, exist_ok=True)
    
    pcap_files = list(samsung_dir.glob("*.pcap")) + list(samsung_dir.glob("*.pcapng"))
    
    output_csv = PROCESSED_DIR / "samsung_s23_flows.csv"
    
    if not pcap_files:
        logger.warning(f"⚠️ No PCAP files found in {samsung_dir}")
        logger.info("Please capture traffic on your S23 and save files with names like:")
        logger.info("- youtube_4k.pcap\n- bgmi_gaming.pcap\n- whatsapp_call.pcap")
        
        # Generate a dummy placeholder so the pipeline doesn't break
        logger.info("Generating a synthetic placeholder dataset for immediate testing...")
        dummy_df = pd.DataFrame(np.random.rand(100, 60), columns=[f"feat_{i}" for i in range(60)])
        dummy_df["label"] = np.random.choice(["gaming", "streaming", "voip"], 100)
        dummy_df["source_device"] = "Samsung Galaxy S23 Ultra (Synthetic)"
        dummy_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Generated synthetic dataset at {output_csv}")
        return

    all_dfs = []
    for pcap in pcap_files:
        # Auto-detect label from filename
        assigned_label = "browsing" # default
        for key, lbl in LABEL_MAP.items():
            if key in pcap.name.lower():
                assigned_label = lbl
                break
                
        df = process_samsung_pcap(pcap, assigned_label)
        if not df.empty:
            all_dfs.append(df)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Successfully integrated {len(final_df)} Samsung S23 flows into {output_csv}")
        
        # Optional: Automatically evaluate the S23 dataset on the trained model
        logger.info("To evaluate this data against your model, you can run your eval script.")
    else:
        logger.error("No valid flows were extracted from the PCAPs.")

if __name__ == "__main__":
    main()
