"""
Task 5.2.2 (threshold tuning): Evaluate different (tp, tn) combinations.

This script sweeps over a grid of positive/negative IoU thresholds and
evaluates each combination on the TRAINING proposals. We also cross-validate
against the VALIDATION split to make sure the chosen thresholds generalise.

Evaluation criteria:
    1. GT Recall     -- fraction of GT boxes covered by >= 1 positive proposal.
    2. Mean Pos IoU  -- average IoU of positive samples (quality of positives).
    3. Sufficiency   -- whether we have enough positives for SVM training.
    4. Gap quality   -- separation between tp and tn (cleaner class boundaries).

Composite score (weighted sum, all terms in [0, 1]):
    score = 0.35 * GT_Recall
          + 0.30 * Mean_Pos_IoU
          + 0.20 * min(n_pos / 200, 1)     (sufficiency)
          + 0.15 * min((tp - tn) / 0.4, 1) (gap quality)

Usage:
    python tune_thresholds.py

Output:
    Prints a ranked table for both train and valid, and the recommended (tp, tn).
"""

import os
import json
import numpy as np

from create_samples import compute_iou, compute_max_iou


# Paths
DATASET_DIR = os.path.join('..', 'data', 'balloon_dataset')
RESULTS_DIR = os.path.join('..', 'results')


def load_data(split_name):
    """Load proposals and COCO annotations for a given split."""
    proposals_path = os.path.join(RESULTS_DIR,
                                  f'proposals_{split_name}.json')
    with open(proposals_path, 'r') as f:
        proposals = json.load(f)

    ann_path = os.path.join(DATASET_DIR, split_name,
                            '_annotations.coco.json')
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    return proposals, coco


def build_gt_per_image(coco):
    """Return dict: filename -> list of [x, y, w, h] GT boxes."""
    id_to_fname = {img['id']: img['file_name'] for img in coco['images']}
    gt = {}
    for ann in coco['annotations']:
        fname = id_to_fname[ann['image_id']]
        gt.setdefault(fname, []).append(ann['bbox'])
    return gt


def evaluate_thresholds(proposals, coco, tp, tn):
    """
    Evaluate a single (tp, tn) setting on one split.

    Returns
    -------
    dict with keys: tp, tn, n_pos, n_neg, n_discarded, neg_pos_ratio,
                    gt_recall, mean_pos_iou, score
    """
    gt_per_image = build_gt_per_image(coco)

    n_pos = 0
    n_neg = 0
    n_discarded = 0
    pos_ious = []

    # Track which GT boxes are recalled (covered by >= 1 positive)
    gt_recalled = {}

    for filename, boxes in proposals.items():
        gt_boxes = gt_per_image.get(filename, [])

        for box in boxes:
            ious = [compute_iou(box, gt) for gt in gt_boxes]
            max_iou = max(ious) if len(ious) > 0 else 0.0

            if max_iou >= tp:
                n_pos += 1
                pos_ious.append(max_iou)
                for gi, iou_val in enumerate(ious):
                    if iou_val >= tp:
                        key = (filename, gi)
                        if key not in gt_recalled or iou_val > gt_recalled[key]:
                            gt_recalled[key] = iou_val
            elif max_iou <= tn:
                n_neg += 1
            else:
                n_discarded += 1

    total_gt = sum(len(v) for v in gt_per_image.values())
    gt_recall = len(gt_recalled) / total_gt if total_gt > 0 else 0.0
    neg_pos_ratio = n_neg / n_pos if n_pos > 0 else float('inf')
    mean_pos_iou = float(np.mean(pos_ious)) if len(pos_ious) > 0 else 0.0

    # Composite score -- weighted sum of four normalised objectives
    #   1) GT recall:     higher is better (can we find the objects?)
    #   2) Mean pos IoU:  higher is better (are positives tight?)
    #   3) Sufficiency:   need enough positives for SVM (target ~200)
    #   4) Gap quality:   wider tp-tn gap means cleaner class separation
    sufficiency = min(n_pos / 200.0, 1.0)
    gap_quality = min((tp - tn) / 0.4, 1.0)

    score = (0.35 * gt_recall
             + 0.30 * mean_pos_iou
             + 0.20 * sufficiency
             + 0.15 * gap_quality)

    return {
        'tp': tp, 'tn': tn,
        'n_pos': n_pos, 'n_neg': n_neg, 'n_discarded': n_discarded,
        'neg_pos_ratio': round(neg_pos_ratio, 2),
        'gt_recall': round(gt_recall, 4),
        'mean_pos_iou': round(mean_pos_iou, 4),
        'score': round(score, 4),
    }


def print_table(results, title):
    """Print a formatted results table."""
    header = (f"{'tp':>5}  {'tn':>5}  {'#pos':>6}  {'#neg':>6}  {'#disc':>6}  "
              f"{'neg/pos':>8}  {'GT_Rec':>7}  {'MeanIoU':>8}  {'Score':>7}")
    print(f"\n{title}")
    print(header)
    print("-" * len(header))

    for r in results:
        ratio_str = f"{r['neg_pos_ratio']:8.1f}" if r['neg_pos_ratio'] < 9999 else "     inf"
        line = (f"{r['tp']:5.2f}  {r['tn']:5.2f}  {r['n_pos']:6d}  "
                f"{r['n_neg']:6d}  {r['n_discarded']:6d}  "
                f"{ratio_str}  {r['gt_recall']:7.4f}  "
                f"{r['mean_pos_iou']:8.4f}  {r['score']:7.4f}")
        print(line)


def main():
    # Load both splits
    print("Loading proposals and annotations ...")
    train_proposals, train_coco = load_data('train')
    valid_proposals, valid_coco = load_data('valid')

    # Threshold grid
    tp_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
    tn_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    train_results = []
    valid_results = []

    for tp in tp_values:
        for tn in tn_values:
            if tn >= tp:
                continue
            train_results.append(
                evaluate_thresholds(train_proposals, train_coco, tp, tn))
            valid_results.append(
                evaluate_thresholds(valid_proposals, valid_coco, tp, tn))

    # Sort both by train score
    paired = list(zip(train_results, valid_results))
    paired.sort(key=lambda p: p[0]['score'], reverse=True)
    train_results = [p[0] for p in paired]
    valid_results = [p[1] for p in paired]

    # Print tables
    print_table(train_results, "=== TRAINING SET ===")
    print_table(valid_results, "=== VALIDATION SET (cross-check) ===")

    # Best by train score
    best_train = train_results[0]
    # Find corresponding valid result
    best_valid = valid_results[0]

    print("\n" + "=" * 60)
    print("Recommended thresholds:  "
          f"tp = {best_train['tp']},  tn = {best_train['tn']}")
    print(f"  TRAIN  ->  GT Recall={best_train['gt_recall']}, "
          f"Positives={best_train['n_pos']}, "
          f"MeanIoU={best_train['mean_pos_iou']}, "
          f"Neg/Pos={best_train['neg_pos_ratio']}")
    print(f"  VALID  ->  GT Recall={best_valid['gt_recall']}, "
          f"Positives={best_valid['n_pos']}, "
          f"MeanIoU={best_valid['mean_pos_iou']}, "
          f"Neg/Pos={best_valid['neg_pos_ratio']}")
    print("=" * 60)

    print(f"\nUse these in create_samples.py:")
    print(f"  python create_samples.py --tp {best_train['tp']} --tn {best_train['tn']}")


if __name__ == '__main__':
    main()
