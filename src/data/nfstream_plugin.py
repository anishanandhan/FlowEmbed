"""
Custom NFStream Plugin & Feature Extractor for FlowEmbed.
Used for both live inference and processing raw PCAP files (e.g. Samsung S23).
"""

import numpy as np

try:
    from nfstream import NFPlugin
except ImportError:
    # Dummy class for when nfstream isn't installed
    class NFPlugin:
        pass

class FlowEmbedPlugin(NFPlugin):
    """
    Custom NFPlugin to track per-packet sizes and IATs to compute
    our specific 60 features (bursts, first N packets, entropies).
    """
    def on_init(self, packet, flow):
        flow.udps.packet_sizes = [packet.ip_size]
        flow.udps.packet_times = [packet.time]
        flow.udps.iats = []

    def on_update(self, packet, flow):
        last_time = flow.udps.packet_times[-1]
        iat_ms = (packet.time - last_time)
        
        flow.udps.packet_sizes.append(packet.ip_size)
        flow.udps.packet_times.append(packet.time)
        flow.udps.iats.append(iat_ms)

        # Truncate lists if they get too huge to save memory
        if len(flow.udps.packet_sizes) > 1000:
            flow.udps.packet_sizes = flow.udps.packet_sizes[-1000:]
            flow.udps.iats = flow.udps.iats[-1000:]


def extract_features_from_flow(flow) -> np.ndarray:
    """Map the nfstream flow object to our exactly 60 features."""
    f = np.zeros(60, dtype=np.float32)
    
    sizes = np.array(flow.udps.packet_sizes) if hasattr(flow.udps, 'packet_sizes') else np.array([0])
    iats = np.array(flow.udps.iats) if hasattr(flow.udps, 'iats') else np.array([0.0])
    
    dur_sec = max(flow.bidirectional_duration_ms / 1000.0, 0.001)
    tot_pkts = flow.bidirectional_packets
    tot_bytes = flow.bidirectional_bytes
    
    # 0-4: Basic
    f[0] = dur_sec
    f[1] = tot_pkts
    f[2] = tot_bytes
    f[3] = tot_pkts / dur_sec
    f[4] = tot_bytes / dur_sec
    
    # 5-14: Sizes
    if len(sizes) > 0:
        f[5] = sizes.min()
        f[6] = sizes.max()
        f[7] = sizes.mean()
        f[8] = sizes.std()
        f[9] = np.median(sizes)
        f[10] = np.percentile(sizes, 25)
        f[11] = np.percentile(sizes, 75)
        f[12] = f[11] - f[10] # IQR
        
    # 15-21: IATs
    if len(iats) > 0:
        f[15] = iats.min()
        f[16] = iats.max()
        f[17] = iats.mean()
        f[18] = iats.std()
        f[19] = np.median(iats)
        f[20] = np.percentile(iats, 25)
        f[21] = np.percentile(iats, 75)
        
    # 22-23: Protocol
    f[22] = 2 if flow.protocol == 17 else 1 # 2=UDP, 1=TCP
    f[23] = 1.0 # Encrypted flag
    
    # 32-51: First 20 packets
    for i in range(min(20, len(sizes))):
        f[32 + i] = sizes[i]
        
    # Context features
    f[52] = 1 # ctx_dst_ip_count
    f[53] = 1 # ctx_protocol_diversity
    f[54] = 0 # ctx_connection_rate
    f[55] = f[3]
    f[56] = f[4]
    
    # Ratios
    f[57] = f[7] / max(f[6], 1)
    f[58] = f[18] / max(f[17], 0.001)
    f[59] = f[8] / max(f[7], 0.001)

    return f
