"""
Evaluation Script — Compute all KPIs and generate results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --benchmark-latency
    python scripts/evaluate.py --holdout-class xr_ar --few-shot 5
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEVICE, CHECKPOINTS_DIR, SPLITS_DIR, EMBEDDINGS_DIR, TRAFFIC_CLASSES,
)
from src.data.dataset import load_split
from src.models.encoder import get_encoder
from src.classifier.knn_classifier import FAISSKNNClassifier
from src.classifier.svm_classifier import SVMClassifier
from src.classifier.evaluator import (
    compute_all_kpis, compute_generalization_metrics, benchmark_latency,
)
from src.explainability.umap_visualizer import compute_umap, plot_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate flow encoder and classifier")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_DIR / "best_encoder.pt"))
    parser.add_argument("--test-data", type=str, default=str(SPLITS_DIR / "test.csv"))
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument("--holdout-class", type=str, default=None)
    parser.add_argument("--few-shot", type=int, default=5)
    parser.add_argument("--save-umap", action="store_true", default=True)
    parser.add_argument("--output", type=str, default=str(Path(__file__).parent.parent / "docs" / "kpi_results.json"))
    return parser.parse_args()


def evaluate(args):
    """Full evaluation pipeline."""
    logger.info("📊 Evaluation starting...")

    # Load encoder
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    encoder = get_encoder(
        encoder_type=checkpoint.get("encoder_type", "transformer"),
        input_dim=checkpoint.get("input_dim", 40),
        embedding_dim=checkpoint.get("embedding_dim", 256),
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(DEVICE)
    encoder.eval()

    # Load test data
    X_test, y_test = load_split(args.test_data)
    logger.info(f"Test set: {len(X_test)} samples, {len(np.unique(y_test))} classes")

    # Generate test embeddings
    test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    batch_size = 1024
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(test_tensor), batch_size):
            batch = test_tensor[i:i+batch_size]
            emb = encoder(batch)
            emb = torch.nn.functional.normalize(emb, dim=1)
            all_embeddings.append(emb.cpu().numpy())

    test_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(EMBEDDINGS_DIR / "test_embeddings.npy", test_embeddings)

    # Load k-NN classifier
    knn = FAISSKNNClassifier()
    knn.load()

    # Classify
    predictions, confidences = knn.predict(test_embeddings)

    # Load label encoder for class names
    import joblib
    le_path = SPLITS_DIR / "label_encoder.pkl"
    label_names = None
    if le_path.exists():
        le = joblib.load(le_path)
        label_names = list(le.classes_)

    # Compute all KPIs
    kpi_results = compute_all_kpis(
        embeddings=test_embeddings,
        labels=y_test,
        y_pred=predictions,
        label_names=label_names,
        encoder=encoder if args.benchmark_latency else None,
        sample_input=test_tensor[:10] if args.benchmark_latency else None,
        faiss_classifier=knn if args.benchmark_latency else None,
        device=str(DEVICE),
    )

    # Print results
    logger.info("=" * 60)
    logger.info("📋 KPI RESULTS")
    logger.info("=" * 60)
    logger.info(f"Intra-class cosine: {kpi_results['intra_class_cosine']['average']:.4f} "
                f"({'✅ PASS' if kpi_results['intra_class_cosine']['pass'] else '❌ FAIL'})")
    logger.info(f"Inter-class cosine: {kpi_results['inter_class_cosine']['average']:.4f} "
                f"({'✅ PASS' if kpi_results['inter_class_cosine']['pass'] else '❌ FAIL'})")
    logger.info(f"Accuracy: {kpi_results['classification']['accuracy']:.4f} "
                f"({'✅ PASS' if kpi_results['classification']['pass'] else '❌ FAIL'})")
    if 'latency' in kpi_results:
        logger.info(f"Latency: {kpi_results['latency']['total_latency_ms']:.2f}ms "
                    f"({'✅ PASS' if kpi_results['latency']['meets_kpi'] else '❌ FAIL'})")
    logger.info(f"Overall: {kpi_results['summary']['kpis_passed']}/{kpi_results['summary']['total_kpis']} KPIs passed")

    # SVM comparison
    logger.info("\n📈 SVM Comparison:")
    svm = SVMClassifier()
    train_embeddings = np.load(EMBEDDINGS_DIR / "train_embeddings.npy")
    train_labels = np.load(EMBEDDINGS_DIR / "train_labels.npy")
    svm.fit(train_embeddings, train_labels)
    svm_preds, svm_confs = svm.predict(test_embeddings)
    from sklearn.metrics import accuracy_score
    svm_acc = accuracy_score(y_test, svm_preds)
    logger.info(f"SVM Accuracy: {svm_acc:.4f}")
    svm.save()

    # UMAP visualization
    if args.save_umap:
        logger.info("\n🎨 Generating UMAP visualization...")

        # Combine train + test for full visualization
        combined_emb = np.concatenate([train_embeddings, test_embeddings])
        combined_labels = np.concatenate([train_labels, y_test])

        # Subsample if too large
        if len(combined_emb) > 10000:
            indices = np.random.choice(len(combined_emb), 10000, replace=False)
            combined_emb = combined_emb[indices]
            combined_labels = combined_labels[indices]

        coords = compute_umap(combined_emb)

        # Save for API
        np.save(EMBEDDINGS_DIR / "umap_coords.npy", coords)
        np.save(EMBEDDINGS_DIR / "umap_labels.npy", combined_labels)

        # Generate plot
        label_map = {i: name for i, name in enumerate(label_names)} if label_names else TRAFFIC_CLASSES
        plot_embeddings(
            coords, combined_labels,
            label_names=label_map,
            save_path=str(Path(__file__).parent.parent / "docs" / "umap_embeddings.png"),
        )
        logger.info("UMAP plot saved")

    # Few-shot generalization test
    if args.holdout_class:
        logger.info(f"\n🔬 Few-shot generalization test (holdout: {args.holdout_class})...")
        # This would use the holdout data prepared during preprocessing
        holdout_path = EMBEDDINGS_DIR / f"holdout_{args.holdout_class}_embeddings.npy"
        if holdout_path.exists():
            holdout_emb = np.load(holdout_path)
            holdout_labels = np.load(holdout_path.replace("embeddings", "labels"))

            # Use only `few_shot` examples as reference, test on rest
            ref_emb = holdout_emb[:args.few_shot]
            ref_labels = holdout_labels[:args.few_shot]
            test_holdout = holdout_emb[args.few_shot:]
            test_holdout_labels = holdout_labels[args.few_shot:]

            # Add reference to k-NN
            knn.add_embeddings(ref_emb, ref_labels)
            preds, _ = knn.predict(test_holdout)

            gen_metrics = compute_generalization_metrics(
                test_holdout_labels, preds, args.holdout_class
            )
            kpi_results["generalization"] = gen_metrics

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(kpi_results, f, indent=2, default=convert_numpy)

    logger.info(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
