import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SPLITS_DIR = ROOT_DIR / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# EXACT 60 Features
FEATURE_COLS = [
    "flow_duration_sec", "total_packets", "total_bytes", "packets_per_sec", "bytes_per_sec",
    "pkt_size_min", "pkt_size_max", "pkt_size_mean", "pkt_size_std", "pkt_size_median",
    "pkt_size_q25", "pkt_size_q75", "pkt_size_iqr", "pkt_size_skewness", "pkt_size_kurtosis",
    "iat_min_ms", "iat_max_ms", "iat_mean_ms", "iat_std_ms", "iat_median_ms",
    "iat_q25_ms", "iat_q75_ms",
    "protocol_encoded", "is_encrypted", "pkt_size_entropy", "iat_entropy",
    "burst_count", "burst_mean_size", "burst_mean_duration_ms", "burst_rate",
    "inter_burst_mean_ms", "burst_bytes_ratio",
    "first_pkt_0", "first_pkt_1", "first_pkt_2", "first_pkt_3", "first_pkt_4",
    "first_pkt_5", "first_pkt_6", "first_pkt_7", "first_pkt_8", "first_pkt_9",
    "first_pkt_10", "first_pkt_11", "first_pkt_12", "first_pkt_13", "first_pkt_14",
    "first_pkt_15", "first_pkt_16", "first_pkt_17", "first_pkt_18", "first_pkt_19",
    "ctx_dst_ip_count", "ctx_protocol_diversity", "ctx_connection_rate",
    "ctx_packets_per_sec", "ctx_bytes_per_sec",
    "avg_payload_ratio", "iat_coefficient_of_variation", "pkt_size_coefficient_of_variation",
]

TARGET_CLASSES = ["gaming", "malware", "streaming", "voip"]

def load_and_align(csv_path):
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        df = df[df["label"].isin(TARGET_CLASSES)]
    if len(df) == 0:
        return None
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    result = df[FEATURE_COLS + ["label"]].copy()
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return result

def main():
    print("=" * 60)
    print("🔧 MASTER DATASET MERGE & SPLIT (CROSS-VALIDATION)")
    print("=" * 60)
    
    print("\n📥 Loading datasets...")
    cesnet = load_and_align(PROCESSED_DIR / "cesnet_flows_processed.csv")
    kaggle = load_and_align(PROCESSED_DIR / "kaggle_5g_flows.csv")
    andmal = load_and_align(PROCESSED_DIR / "kaggle_andmal_flows.csv")
    mirage = load_and_align(PROCESSED_DIR / "kaggle_mirage_flows.csv")
    
    print("\n🔀 Merging all datasets...")
    all_data = pd.concat([cesnet, kaggle, andmal, mirage], ignore_index=True)
    print(f"  Total combined flows: {len(all_data):,}")
    
    # Global 80/20 split for Cross Validation!
    train_full, test_full = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data["label"])
    
    print("\n⚖️  Balancing training classes...")
    balanced_train = []
    train_targets = {"gaming": 15000, "malware": 12000, "streaming": 20000, "voip": 5000}
    
    for label in TARGET_CLASSES:
        df_class = train_full[train_full["label"] == label]
        target = train_targets.get(label, len(df_class))
        if len(df_class) > target:
            df_class = df_class.sample(n=target, random_state=42)
        elif len(df_class) > 0 and len(df_class) < target:
            df_class = df_class.sample(n=target, replace=True, random_state=42)
        balanced_train.append(df_class)

    train_balanced = pd.concat(balanced_train, ignore_index=True).sample(frac=1.0, random_state=42)
    
    print("\n🏷️  Creating label encoder...")
    le = LabelEncoder()
    le.fit(TARGET_CLASSES)
    train_balanced["label"] = le.transform(train_balanced["label"])
    test_full["label"] = le.transform(test_full["label"])
    
    train_df, val_df = train_test_split(train_balanced, test_size=0.15, random_state=42, stratify=train_balanced["label"])
    
    print("\n📐 Scaling features...")
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[FEATURE_COLS])
    val_features = scaler.transform(val_df[FEATURE_COLS])
    test_features = scaler.transform(test_full[FEATURE_COLS])
    
    train_scaled = pd.DataFrame(train_features, columns=FEATURE_COLS)
    train_scaled["label"] = train_df["label"].values
    val_scaled = pd.DataFrame(val_features, columns=FEATURE_COLS)
    val_scaled["label"] = val_df["label"].values
    test_scaled = pd.DataFrame(test_features, columns=FEATURE_COLS)
    test_scaled["label"] = test_full["label"].values
    
    print("\n💾 Saving splits...")
    train_scaled.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_scaled.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_scaled.to_csv(SPLITS_DIR / "test.csv", index=False)
    
    joblib.dump(scaler, SPLITS_DIR / "scaler.pkl")
    joblib.dump(le, SPLITS_DIR / "label_encoder.pkl")
    
    print(f"\n✅ COMPLETE! Train: {len(train_scaled):,}, Val: {len(val_scaled):,}, Test: {len(test_scaled):,}")

if __name__ == "__main__":
    main()
