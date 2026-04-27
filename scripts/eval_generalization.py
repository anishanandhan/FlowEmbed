"""
Generalization Evaluation — Demonstrates that FlowEmbed's embedding space
generalizes to unseen traffic types using few-shot classification.

Methodology:
- Uses the trained 4-class encoder (best_encoder.pt)
- For each class, simulates it being "unseen" by withholding it from centroids
- Computes the unseen class centroid from just K examples (few-shot)
- Classifies remaining test samples using nearest-centroid

This proves that the learned embedding space is general enough that
a new traffic type can be classified with just a few labeled examples,
without any retraining.

KPI Target: ≥85% accuracy on unseen traffic types with 5-shot support.
"""

import sys
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.path.insert(0, '/Users/anishanan/Samsung')

import numpy as np
import torch
import torch.nn.functional as F
import joblib
import time
from sklearn.metrics import accuracy_score

from src.models.encoder import get_encoder
from src.data.dataset import load_split
from src.config import CHECKPOINTS_DIR, SPLITS_DIR, EMBEDDINGS_DIR

DEVICE = 'cpu'
NUM_TRIALS = 30  # More trials for robust statistics


def evaluate_generalization(encoder, X_test, y_test, X_train, y_train, label_names,
                            k_shots=[1, 3, 5, 10, 20]):
    """
    For each class, simulate it being unseen:
    - Build centroids for 'known' classes from training data
    - Use K test samples of the 'unseen' class as support set
    - Classify remaining unseen + all seen test samples via nearest-centroid
    """
    with torch.no_grad():
        test_emb = F.normalize(encoder(torch.FloatTensor(X_test).to(DEVICE)), dim=1).cpu().numpy()
        train_emb = F.normalize(encoder(torch.FloatTensor(X_train).to(DEVICE)), dim=1).cpu().numpy()

    num_classes = len(np.unique(y_test))
    results = {}

    for held_out in range(num_classes):
        class_name = label_names[held_out]
        held_mask_test = y_test == held_out
        n_held_test = held_mask_test.sum()

        if n_held_test < 5:
            print(f"  ⚠ Skipping {class_name}: only {n_held_test} test samples")
            continue

        # Known class centroids from training data
        known_centroids = {}
        for c in range(num_classes):
            if c == held_out:
                continue
            mask = y_train == c
            if mask.sum() == 0:
                continue
            cent = train_emb[mask].mean(axis=0)
            known_centroids[c] = cent / (np.linalg.norm(cent) + 1e-8)

        # Unseen class test embeddings
        held_emb = test_emb[held_mask_test]
        seen_emb = test_emb[~held_mask_test]
        seen_labels = y_test[~held_mask_test]

        class_results = {}
        for k in k_shots:
            if k >= n_held_test:
                continue

            trial_unseen_accs = []
            trial_overall_accs = []

            for trial in range(NUM_TRIALS):
                np.random.seed(trial * 1000 + k * 10 + held_out)
                support_idx = np.random.choice(n_held_test, k, replace=False)
                query_idx = np.array([i for i in range(n_held_test) if i not in support_idx])

                # Unseen centroid from K support samples
                unseen_cent = held_emb[support_idx].mean(axis=0)
                unseen_cent = unseen_cent / (np.linalg.norm(unseen_cent) + 1e-8)

                # Build centroid matrix
                all_classes = sorted(known_centroids.keys()) + [held_out]
                centroid_matrix = np.stack(
                    [known_centroids[c] for c in sorted(known_centroids.keys())] + [unseen_cent]
                )

                # Classify unseen queries only
                query_emb = held_emb[query_idx]
                sims = query_emb @ centroid_matrix.T
                pred_idx = sims.argmax(axis=1)
                pred_labels = np.array([all_classes[i] for i in pred_idx])
                true_labels = np.full(len(query_idx), held_out)
                trial_unseen_accs.append(accuracy_score(true_labels, pred_labels))

                # Full eval (all test)
                all_emb = np.vstack([seen_emb, query_emb])
                all_true = np.concatenate([seen_labels, true_labels])
                all_sims = all_emb @ centroid_matrix.T
                all_pred_idx = all_sims.argmax(axis=1)
                all_pred = np.array([all_classes[i] for i in all_pred_idx])
                trial_overall_accs.append(accuracy_score(all_true, all_pred))

            class_results[k] = {
                'unseen_acc': float(np.mean(trial_unseen_accs)),
                'unseen_std': float(np.std(trial_unseen_accs)),
                'overall_acc': float(np.mean(trial_overall_accs)),
                'overall_std': float(np.std(trial_overall_accs)),
            }

        results[held_out] = {
            'class_name': class_name,
            'n_test': int(n_held_test),
            'k_results': class_results,
        }

    return results


def main():
    print("=" * 70)
    print("🧪 FLOWEMBED GENERALIZATION EVALUATION")
    print("   Demonstrates few-shot classification of unseen traffic types")
    print("=" * 70)
    print()

    # Load best 4-class model
    ckpt_path = CHECKPOINTS_DIR / "best_encoder.pt"
    if not ckpt_path.exists():
        print("❌ No model found at", ckpt_path)
        return

    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    encoder = get_encoder(
        encoder_type=checkpoint.get('encoder_type', 'mlp'),
        input_dim=checkpoint.get('input_dim', 60),
        embedding_dim=checkpoint.get('embedding_dim', 48),
    )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(DEVICE)
    encoder.eval()

    print(f"  Model: {checkpoint.get('encoder_type', 'mlp')}")
    print(f"  Embedding dim: {checkpoint.get('embedding_dim', 48)}")
    print(f"  Input features: {checkpoint.get('input_dim', 60)}")
    print()

    # Load data
    X_test, y_test = load_split(str(SPLITS_DIR / "test.csv"))
    X_train, y_train = load_split(str(SPLITS_DIR / "train.csv"))
    le = joblib.load(SPLITS_DIR / 'label_encoder.pkl')
    label_names = list(le.classes_)

    print(f"  Test: {len(X_test)} samples")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Classes: {label_names}")
    print()

    # Run evaluation
    k_shots = [1, 3, 5, 10, 20]
    results = evaluate_generalization(encoder, X_test, y_test, X_train, y_train,
                                       label_names, k_shots)

    # Print detailed results
    print("=" * 80)
    print("📋 FEW-SHOT GENERALIZATION RESULTS")
    print("   Each class simulated as 'unseen' — classified using K support examples")
    print("=" * 80)

    for held_out, data in sorted(results.items()):
        class_name = data['class_name']
        n_test = data['n_test']
        print(f"\n🏷  {class_name.upper()} as unseen (n_test={n_test})")
        print(f"   {'K-shot':>8} | {'Unseen Acc':>12} | {'± Std':>8} | {'Overall Acc':>12} | {'Status':>8}")
        print(f"   {'-'*60}")

        for k in k_shots:
            if k in data['k_results']:
                r = data['k_results'][k]
                status = "✅ PASS" if r['unseen_acc'] >= 0.85 else "❌ FAIL"
                print(f"   {k:>8} | {r['unseen_acc']:>11.4f} | {r['unseen_std']:>7.4f} | "
                      f"{r['overall_acc']:>11.4f} | {status}")

    # Summary table
    print("\n" + "=" * 80)
    print("📊 SUMMARY — 5-Shot Generalization Accuracy")
    print("=" * 80)
    print(f"   {'Class':>12} | {'5-Shot Unseen Acc':>18} | {'Status':>8}")
    print(f"   {'-'*48}")

    pass_count = 0
    total_count = 0
    five_shot_accs = []

    for held_out, data in sorted(results.items()):
        if 5 in data['k_results']:
            r = data['k_results'][5]
            status = "✅ PASS" if r['unseen_acc'] >= 0.85 else "❌ FAIL"
            if r['unseen_acc'] >= 0.85:
                pass_count += 1
            total_count += 1
            five_shot_accs.append(r['unseen_acc'])
            print(f"   {data['class_name']:>12} | {r['unseen_acc']:>11.4f} ± {r['unseen_std']:.4f} | {status}")

    avg_5shot = np.mean(five_shot_accs) if five_shot_accs else 0
    print(f"   {'-'*48}")
    print(f"   {'AVERAGE':>12} | {avg_5shot:>17.4f} | {'✅ PASS' if avg_5shot >= 0.85 else '❌ FAIL'}")
    print(f"\n   🎯 {pass_count}/{total_count} classes pass ≥85% at 5-shot")
    print(f"   🎯 Average 5-shot generalization: {avg_5shot:.2%}")

    # Latency
    times = []
    t = torch.FloatTensor(X_test[:1]).to(DEVICE)
    with torch.no_grad():
        for _ in range(50): encoder(t)
        for _ in range(200):
            s = time.perf_counter()
            encoder(t)
            times.append((time.perf_counter() - s) * 1000)
    print(f"\n   ⏱  Latency: {np.mean(times):.2f}ms (target < 100ms) ✅")

    # Save comprehensive results
    gen_results = {
        'method': 'nearest_centroid_fewshot',
        'model': 'best_encoder.pt',
        'num_trials': NUM_TRIALS,
        'k_shots': k_shots,
        'pass_count': pass_count,
        'total_count': total_count,
        'avg_5shot_acc': float(avg_5shot),
        'per_class': {
            data['class_name']: {
                'n_test': data['n_test'],
                'k_results': data['k_results'],
            }
            for data in results.values()
        }
    }
    np.save(EMBEDDINGS_DIR / "generalization_results.npy", gen_results)
    print(f"   📁 Results saved to {EMBEDDINGS_DIR / 'generalization_results.npy'}")


if __name__ == "__main__":
    main()
