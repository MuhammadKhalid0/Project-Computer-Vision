"""
Task 5.2.2: Create positive and negative training samples from region proposals.

This script loads the pre-computed proposals (from Task 5.2.1), computes the
Intersection-over-Union (IoU) between each proposal and the ground truth
annotations, and labels proposals as:
    - positive (class "balloon") if max IoU >= tp
    - negative (class "background") if max IoU <= tn
    - discarded (ambiguous) if tn < max IoU < tp

The labeled samples are saved to JSON for feature extraction (Task 5.2.2 cont.)
and SVM training (Task 5.2.3).

Usage:
    python create_samples.py [--tp 0.5] [--tn 0.3]

Output:
    results/samples_train.json
    results/samples_valid.json
"""

import os
import json
import argparse
import numpy as np


# Paths
DATASET_DIR = os.path.join('..', 'data', 'balloon_dataset')
RESULTS_DIR = os.path.join('..', 'results')


# -----------------------------------------------------------------------
# IoU computation
# -----------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """
    Compute Intersection-over-Union between two bounding boxes.

    Both boxes are given as [x, y, w, h] (COCO format), where (x, y) is the
    top-left corner.

    Parameters
    ----------
    box_a : list or array of length 4
    box_b : list or array of length 4

    Returns
    -------
    float
        IoU value in [0, 1].
    """
    # Convert [x, y, w, h] -> [x1, y1, x2, y2]
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = box_a[0] + box_a[2], box_a[1] + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = box_b[0] + box_b[2], box_b[1] + box_b[3]

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    # Union
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_max_iou(proposal, gt_boxes):
    """
    Compute the maximum IoU of a single proposal against all GT boxes.

    Parameters
    ----------
    proposal : list
        [x, y, w, h] of the proposal.
    gt_boxes : list of lists
        Each element is [x, y, w, h] of a ground truth box.

    Returns
    -------
    float
        Maximum IoU value (0.0 if gt_boxes is empty).
    """
    if len(gt_boxes) == 0:
        return 0.0
    return max(compute_iou(proposal, gt) for gt in gt_boxes)


# -----------------------------------------------------------------------
# Labelling
# -----------------------------------------------------------------------

def label_proposals(proposals, coco, tp, tn):
    """
    Label proposals as positive or negative based on IoU with GT.

    Parameters
    ----------
    proposals : dict
        {filename: [[x, y, w, h], ...]} from the proposals JSON.
    coco : dict
        Parsed COCO annotation JSON for this split.
    tp : float
        Positive threshold -- proposals with max IoU >= tp are positive.
    tn : float
        Negative threshold -- proposals with max IoU <= tn are negative.

    Returns
    -------
    list of dict
        Each entry: {"filename": str, "box": [x,y,w,h], "label": 0 or 1,
                      "max_iou": float}
    """
    # Build a lookup: image_id -> filename
    id_to_filename = {}
    for img_entry in coco['images']:
        id_to_filename[img_entry['id']] = img_entry['file_name']

    # Build a lookup: filename -> list of GT boxes [x, y, w, h]
    gt_per_image = {}
    for ann in coco['annotations']:
        fname = id_to_filename[ann['image_id']]
        if fname not in gt_per_image:
            gt_per_image[fname] = []
        gt_per_image[fname].append(ann['bbox'])  # already [x, y, w, h]

    samples = []

    for filename, boxes in proposals.items():
        gt_boxes = gt_per_image.get(filename, [])

        for box in boxes:
            max_iou = compute_max_iou(box, gt_boxes)

            if max_iou >= tp:
                # Positive sample (balloon)
                samples.append({
                    "filename": filename,
                    "box": box,
                    "label": 1,
                    "max_iou": round(max_iou, 4),
                })
            elif max_iou <= tn:
                # Negative sample (background)
                samples.append({
                    "filename": filename,
                    "box": box,
                    "label": 0,
                    "max_iou": round(max_iou, 4),
                })
            # else: ambiguous, skip

    return samples


def print_summary(samples, total_proposals, split_name, tp, tn):
    """Print a short summary of the labeled samples."""
    positives = [s for s in samples if s['label'] == 1]
    negatives = [s for s in samples if s['label'] == 0]
    n_discarded = total_proposals - len(positives) - len(negatives)

    print(f"\n{split_name} split (tp={tp}, tn={tn}):")
    print(f"  Positives : {len(positives)}")
    print(f"  Negatives : {len(negatives)}")
    print(f"  Discarded : {n_discarded}")

    if len(positives) > 0:
        ious = [s['max_iou'] for s in positives]
        print(f"  Pos IoU   : mean={np.mean(ious):.3f}, "
              f"min={np.min(ious):.3f}, max={np.max(ious):.3f}")

    if len(positives) + len(negatives) > 0:
        ratio = len(negatives) / max(len(positives), 1)
        print(f"  Neg/Pos   : {ratio:.1f}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create positive/negative samples from proposals + GT.")
    parser.add_argument('--tp', type=float, default=0.5,
                        help='Positive IoU threshold (default: 0.5)')
    parser.add_argument('--tn', type=float, default=0.3,
                        help='Negative IoU threshold (default: 0.3)')
    args = parser.parse_args()

    tp = args.tp
    tn = args.tn
    print(f"Using thresholds: tp={tp}, tn={tn}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for split_name in ['train', 'valid']:
        # Load proposals from Task 5.2.1
        proposals_path = os.path.join(RESULTS_DIR,
                                      f'proposals_{split_name}.json')
        with open(proposals_path, 'r') as f:
            proposals = json.load(f)

        # Load COCO annotations
        split_dir = os.path.join(DATASET_DIR, split_name)
        ann_path = os.path.join(split_dir, '_annotations.coco.json')
        with open(ann_path, 'r') as f:
            coco = json.load(f)

        # Label proposals
        total_proposals = sum(len(v) for v in proposals.values())
        samples = label_proposals(proposals, coco, tp, tn)
        print_summary(samples, total_proposals, split_name, tp, tn)

        # Save to JSON
        out_path = os.path.join(RESULTS_DIR, f'samples_{split_name}.json')
        with open(out_path, 'w') as f:
            json.dump(samples, f)
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

