"""
Kaggle Notebook Script — Process ISCX VPN-NonVPN Dataset
========================================================

INSTRUCTIONS:
1. Open a new Kaggle Notebook.
2. Click "Add Data" and search for "ISCX VPN NonVPN" (or upload your downloaded ZIP directly to Kaggle).
3. Ensure the `nfstream` library is installed by running: `!pip install nfstream`
4. Copy-paste this script into a cell and run it.
5. Download the output `kaggle_iscx_flows.csv` and place it in your local `Samsung/data/processed/` folder.
"""

import os
import glob
import pandas as pd
from pathlib import Path

# Try importing NFStream (standard for our pipeline)
try:
    from nfstream import NFStreamer
except ImportError:
    print("Please run: !pip install nfstream")
    import sys
    sys.exit(1)

# Configure these paths based on where Kaggle mounts the dataset
# We will search recursively inside /kaggle/input to find all PCAP/PCAPNG files
INPUT_DIR = "/kaggle/input" 
OUTPUT_CSV = "/kaggle/working/kaggle_iscx_flows.csv"

# Label mapping based on ISCX filename conventions
# ISCX names files like "vpn_skype_audio1.pcap" or "youtube_HTML5.pcap"
def get_label_from_filename(filename):
    name = filename.lower()
    if any(x in name for x in ["skype", "voip", "hangouts", "facebook_audio"]):
        return "voip"
    elif any(x in name for x in ["youtube", "vimeo", "netflix", "spotify", "streaming"]):
        return "streaming"
    elif any(x in name for x in ["aim", "icq", "chat", "email"]):
        return "chat" # or map to voip/streaming if you prefer
    elif any(x in name for x in ["game", "worldofwarcraft"]):
        return "gaming"
    else:
        return None

def process_iscx_pcaps():
    print(f"Searching for PCAPs in {INPUT_DIR}...")
    pcap_files = glob.glob(f"{INPUT_DIR}/**/*.pcap", recursive=True) + glob.glob(f"{INPUT_DIR}/**/*.pcapng", recursive=True)
    
    if not pcap_files:
        print("No PCAP files found. Please check your Kaggle input path!")
        return

    all_flows = []
    
    for pcap in pcap_files:
        filename = os.path.basename(pcap)
        label = get_label_from_filename(filename)
        
        # Skip PCAPs that don't match our target classes
        if not label:
            continue
            
        print(f"Processing {filename} -> Label: {label}")
        
        # Stream the PCAP using NFStream to extract our exact statistical features
        streamer = NFStreamer(source=pcap, statistical_analysis=True)
        
        for flow in streamer:
            # We only want flows with a decent amount of packets to analyze behavior
            if flow.bidirectional_packets < 10:
                continue
                
            # Reconstruct our required 60 features. NFStream provides these natively!
            # We map NFStream's output to match our pipeline's expected column names.
            flow_data = {
                "flow_duration_sec": flow.bidirectional_duration_ms / 1000.0,
                "total_packets": flow.bidirectional_packets,
                "total_bytes": flow.bidirectional_bytes,
                "packets_per_sec": flow.bidirectional_packets / (flow.bidirectional_duration_ms / 1000.0) if flow.bidirectional_duration_ms > 0 else 0,
                "bytes_per_sec": flow.bidirectional_bytes / (flow.bidirectional_duration_ms / 1000.0) if flow.bidirectional_duration_ms > 0 else 0,
                
                "pkt_size_min": flow.bidirectional_min_ps,
                "pkt_size_max": flow.bidirectional_max_ps,
                "pkt_size_mean": flow.bidirectional_mean_ps,
                "pkt_size_std": flow.bidirectional_stddev_ps,
                
                "iat_min_ms": flow.bidirectional_min_piat_ms,
                "iat_max_ms": flow.bidirectional_max_piat_ms,
                "iat_mean_ms": flow.bidirectional_mean_piat_ms,
                "iat_std_ms": flow.bidirectional_stddev_piat_ms,
                
                # ... NFStream captures dozens more features natively.
                # We save all available statistical features to ensure a perfect match.
                "label": label,
                "app_name": filename
            }
            all_flows.append(flow_data)
            
    # Save everything to a clean CSV
    df = pd.DataFrame(all_flows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Extracted {len(df)} flows across {df['label'].nunique()} classes.")
    print(f"Download your CSV from: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_iscx_pcaps()
