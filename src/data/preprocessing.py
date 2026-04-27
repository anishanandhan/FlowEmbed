"""
Preprocessing — Clean, normalize, and prepare flow features for training.

Handles multiple dataset formats (nfstream output, Kaggle 5G CSV, CESNET-QUIC22).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

from src.config import (
    FLOW_FEATURES, PROCESSED_DIR, SPLITS_DIR, NUM_INPUT_FEATURES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, TRAFFIC_CLASSES,
)

logger = logging.getLogger(__name__)


def load_and_merge_datasets(
    dataset_paths: List[str],
    label_column: str = "label",
) -> pd.DataFrame:
    """
    Load multiple CSV datasets and merge them into a single DataFrame.

    Args:
        dataset_paths: List of paths to CSV files.
        label_column: Name of the label column.

    Returns:
        Merged DataFrame.
    """
    all_data = []

    for path in dataset_paths:
        path = Path(path)
        if not path.exists():
            logger.warning(f"Dataset not found: {path}")
            continue

        logger.info(f"Loading dataset: {path}")
        df = pd.read_csv(path)

        if label_column not in df.columns:
            logger.warning(f"Label column '{label_column}' not found in {path}")
            continue

        all_data.append(df)

    if not all_data:
        raise ValueError("No valid datasets found!")

    merged = pd.concat(all_data, ignore_index=True)
    logger.info(f"Merged {len(merged)} total flows from {len(all_data)} datasets")

    return merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the feature matrix:
    - Remove rows with too many NaN values
    - Impute remaining NaN with median
    - Remove infinite values
    - Remove constant columns
    """
    initial_rows = len(df)

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with >50% NaN
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)

    # Impute remaining NaN with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Remove constant columns (zero variance)
    constant_cols = [col for col in numeric_cols if df[col].std() == 0]
    if constant_cols:
        logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols[:5]}...")
        df = df.drop(columns=constant_cols)

    logger.info(f"Cleaned: {initial_rows} → {len(df)} rows")
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the most relevant flow features for training.

    Tries to use predefined FLOW_FEATURES, falls back to auto-selection
    if the dataset uses different column names.
    """
    # Try predefined features
    available_features = [f for f in FLOW_FEATURES if f in df.columns]

    if len(available_features) >= 20:
        logger.info(f"Using {len(available_features)} predefined features")
        return df[available_features], available_features

    # Fallback: auto-select numeric features (excluding IPs, ports, timestamps)
    exclude_patterns = [
        "ip", "port", "mac", "timestamp", "id", "expiration", "label",
        "source_file", "application_name", "application_category",
        "requested_server_name", "client_fingerprint", "server_fingerprint",
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected = []
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(pat in col_lower for pat in exclude_patterns):
            selected.append(col)

    # Limit to top N by variance
    if len(selected) > NUM_INPUT_FEATURES:
        variances = df[selected].var().sort_values(ascending=False)
        selected = variances.head(NUM_INPUT_FEATURES).index.tolist()

    logger.info(f"Auto-selected {len(selected)} features")
    return df[selected], selected


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Apply MinMaxScaler normalization. Fit on train, transform all splits.
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if save_path:
        joblib.dump(scaler, save_path)
        logger.info(f"Saved scaler to {save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def encode_labels(
    y: pd.Series,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if save_path:
        joblib.dump(le, save_path)
        logger.info(f"Saved label encoder to {save_path}")

    # Log class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    for cls_id, count in zip(unique, counts):
        cls_name = le.inverse_transform([cls_id])[0]
        logger.info(f"  Class {cls_id} ({cls_name}): {count} samples")

    return y_encoded, le


def prepare_dataset(
    dataset_paths: List[str],
    label_column: str = "label",
    holdout_class: Optional[str] = None,
) -> dict:
    """
    Full preprocessing pipeline: load → clean → select → split → normalize.

    Args:
        dataset_paths: List of paths to CSV files.
        label_column: Name of the label column.
        holdout_class: If set, hold out this class entirely for few-shot testing.

    Returns:
        Dictionary with all processed data, scalers, and encoders.
    """
    # Load and merge
    df = load_and_merge_datasets(dataset_paths, label_column)

    # Clean
    df = clean_features(df)

    # Separate holdout class if specified
    holdout_data = None
    if holdout_class and holdout_class in df[label_column].values:
        holdout_mask = df[label_column] == holdout_class
        holdout_data = df[holdout_mask].copy()
        df = df[~holdout_mask].copy()
        logger.info(f"Held out {len(holdout_data)} flows of class '{holdout_class}' for few-shot testing")

    # Select features
    X_df, feature_names = select_features(df)
    y = df[label_column]

    # Encode labels
    y_encoded, label_encoder = encode_labels(
        y, save_path=str(SPLITS_DIR / "label_encoder.pkl")
    )

    # Train/val/test split (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df.values, y_encoded,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y_encoded,
        random_state=42,
    )

    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=42,
    )

    # Normalize
    X_train, X_val, X_test, scaler = normalize_features(
        X_train, X_val, X_test,
        save_path=str(SPLITS_DIR / "scaler.pkl"),
    )

    # Save splits
    _save_split(X_train, y_train, feature_names, SPLITS_DIR / "train.csv")
    _save_split(X_val, y_val, feature_names, SPLITS_DIR / "val.csv")
    _save_split(X_test, y_test, feature_names, SPLITS_DIR / "test.csv")

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "num_classes": len(label_encoder.classes_),
    }

    # Process holdout class if available
    if holdout_data is not None:
        X_holdout, _ = select_features(holdout_data)
        X_holdout_scaled = scaler.transform(X_holdout.values)
        result["X_holdout"] = X_holdout_scaled
        result["holdout_class"] = holdout_class

    return result


def _save_split(X: np.ndarray, y: np.ndarray, feature_names: list, path: Path):
    """Save a data split to CSV."""
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} samples to {path}")


def load_kaggle_5g(csv_path: str, label_column: str = "label") -> pd.DataFrame:
    """
    Load and standardize the Kaggle 5G Traffic dataset.
    Adapts column names to match our feature naming convention.
    """
    df = pd.read_csv(csv_path)

    # The Kaggle 5G dataset may use different column names
    # This mapping handles common variations
    column_mapping = {
        "Flow Duration": "bidirectional_duration_ms",
        "Total Fwd Packets": "src2dst_packets",
        "Total Backward Packets": "dst2src_packets",
        "Fwd Packet Length Mean": "src2dst_mean_ps",
        "Bwd Packet Length Mean": "dst2src_mean_ps",
        "Fwd Packet Length Std": "src2dst_stddev_ps",
        "Bwd Packet Length Std": "dst2src_stddev_ps",
        "Flow IAT Mean": "bidirectional_mean_piat_ms",
        "Flow IAT Std": "bidirectional_stddev_piat_ms",
        "Fwd IAT Mean": "src2dst_mean_piat_ms",
        "Bwd IAT Mean": "dst2src_mean_piat_ms",
        "Label": label_column,
    }

    # Apply mapping for columns that exist
    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)

    return df
