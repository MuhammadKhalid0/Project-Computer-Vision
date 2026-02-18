"""
Task 5.2.5: Evaluate the detection pipeline on the test set.

Two evaluation metrics are computed:

1. COCO mAP (mean Average Precision):
   Standard object detection metric. We run the full detection pipeline on each
   test image, collect all detections in COCO result format, and use the official
   pycocotools COCOeval to compute AP at multiple IoU thresholds (0.50:0.95).

2. MABO (Mean Average Best Overlap):
   Proposal quality metric from Uijlings et al. For each ground truth box, we
   find the proposal with the highest IoU (Best Overlap). MABO is the average
   of these Best Overlaps across all GT boxes. A high MABO means the proposals
   cover the objects well, regardless of classification quality.

Usage:
    python evaluate.py [--conf 0.5] [--nms 0.3]

Input:
    data/balloon_dataset/test/ (images + annotations.coco.json)
    results/proposals_test.json
    results/svm_model.joblib, results/svm_scaler.joblib
"""

import os
import gc
import json
import argparse
import numpy as np
import skimage.io
import torch
import joblib

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import shared functions from detect.py
from detect import (
    build_feature_extractor,
    crop_region,
    non_maximum_suppression,
)


# Paths
DATASET_DIR = os.path.join('..', 'data', 'balloon_dataset')
RESULTS_DIR = os.path.join('..', 'results')


# ------------------------------------------------------------------
# IoU helper for MABO
# ------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """
    Compute IoU between two [x, y, w, h] boxes.
    """
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter

    if union == 0:
        return 0.0
    return inter / union


# ------------------------------------------------------------------
# Detection on a single image (using pre-loaded proposals)
# ------------------------------------------------------------------

def detect_image(image, proposals, model, preprocess, svm, scaler,
                 conf_threshold=0.8, nms_threshold=0.15):
    """
    Run classification + NMS on pre-computed proposals for one image.

    Uses the saved proposals instead of re-running selective search.

    Returns
    -------
    list of dict
        Each: {'box': [x,y,w,h], 'score': float}
    """
    if len(proposals) == 0:
        return []

    # Extract features in mini-batches to avoid OOM on large proposal sets
    BATCH_SIZE = 64
    all_features = []
    for start in range(0, len(proposals), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(proposals))
        batch_tensors = []
        for box in proposals[start:end]:
            crop = crop_region(image, box)
            tensor = preprocess(crop)
            batch_tensors.append(tensor)
        batch = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = model(batch).numpy()
        all_features.append(feats)
        del batch, batch_tensors
    features = np.concatenate(all_features, axis=0)
    del all_features

    # Classify
    features = scaler.transform(features)
    probabilities = svm.predict_proba(features)
    balloon_scores = probabilities[:, 1]

    # Filter by confidence
    det_boxes = []
    det_scores = []
    for i, score in enumerate(balloon_scores):
        if score >= conf_threshold:
            det_boxes.append(proposals[i])
            det_scores.append(score)

    if len(det_boxes) == 0:
        return []

    det_boxes = np.array(det_boxes, dtype=np.float32)
    det_scores = np.array(det_scores)

    # NMS
    keep = non_maximum_suppression(det_boxes, det_scores, nms_threshold)

    detections = []
    for i in keep:
        detections.append({
            'box': det_boxes[i].tolist(),
            'score': float(det_scores[i]),
        })

    return detections


# ------------------------------------------------------------------
# MABO: Mean Average Best Overlap (Uijlings et al.)
# ------------------------------------------------------------------

def compute_mabo(proposals_dict, coco_data):
    """
    Compute MABO across the test set.

    For each ground truth bounding box, find the proposal with the highest IoU
    (the "Best Overlap"). MABO is the mean of all these Best Overlaps.

    This metric measures the quality of proposals (how well they cover the GT),
    independent of the classifier.

    Parameters
    ----------
    proposals_dict : dict
        filename -> list of [x, y, w, h] proposals.
    coco_data : dict
        Raw COCO annotation JSON.

    Returns
    -------
    mabo : float
        Mean Average Best Overlap.
    best_overlaps : list of float
        Best Overlap for each GT box (for detailed analysis).
    """
    # Build lookup: image_id -> filename
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    best_overlaps = []

    for ann in coco_data['annotations']:
        gt_box = ann['bbox']  # [x, y, w, h]
        image_id = ann['image_id']
        filename = id_to_filename[image_id]

        proposals = proposals_dict.get(filename, [])

        if len(proposals) == 0:
            best_overlaps.append(0.0)
            continue

        # Find the proposal with the highest IoU to this GT box
        best_iou = 0.0
        for prop in proposals:
            iou = compute_iou(gt_box, prop)
            if iou > best_iou:
                best_iou = iou
        best_overlaps.append(best_iou)

    mabo = np.mean(best_overlaps) if len(best_overlaps) > 0 else 0.0
    return mabo, best_overlaps


# ------------------------------------------------------------------
# COCO mAP evaluation
# ------------------------------------------------------------------

def evaluate_map(coco_gt, all_detections, cat_id):
    """
    Evaluate detections using official COCO mAP.

    Parameters
    ----------
    coco_gt : COCO
        Ground truth COCO object.
    all_detections : list of dict
        COCO result format: [{'image_id', 'category_id', 'bbox', 'score'}, ...]
    cat_id : int
        Category ID for "balloon".

    Returns
    -------
    dict
        mAP results at various thresholds.
    """
    if len(all_detections) == 0:
        print("  No detections to evaluate!")
        return {}

    # Save detections to a temp file (COCOeval requires it)
    det_path = os.path.join(RESULTS_DIR, 'coco_detections_test.json')
    with open(det_path, 'w') as f:
        json.dump(all_detections, f)

    coco_dt = coco_gt.loadRes(det_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = [cat_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        'AP_0.50:0.95': coco_eval.stats[0],
        'AP_0.50': coco_eval.stats[1],
        'AP_0.75': coco_eval.stats[2],
        'AR_0.50:0.95': coco_eval.stats[8],
    }
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate detection pipeline on the test set.")
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.3,
                        help='NMS IoU threshold (default: 0.3)')
    args = parser.parse_args()

    test_dir = os.path.join(DATASET_DIR, 'test')
    ann_path = os.path.join(test_dir, '_annotations.coco.json')

    # Load annotations (raw JSON for MABO, COCO object for mAP)
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    print("Loading COCO ground truth ...")
    coco_gt = COCO(ann_path)

    # Determine the balloon category ID used in annotations
    # The dataset has two entries; use the one that actually appears in annotations
    ann_cat_ids = set(a['category_id'] for a in coco_data['annotations'])
    cat_id = list(ann_cat_ids)[0]
    print(f"  Balloon category_id: {cat_id}")

    # Load proposals
    proposals_path = os.path.join(RESULTS_DIR, 'proposals_test.json')
    with open(proposals_path, 'r') as f:
        proposals_dict = json.load(f)
    print(f"  Loaded proposals for {len(proposals_dict)} images")

    # Load SVM model and feature extractor
    print("Loading SVM model ...")
    svm = joblib.load(os.path.join(RESULTS_DIR, 'svm_model.joblib'))
    scaler = joblib.load(os.path.join(RESULTS_DIR, 'svm_scaler.joblib'))

    print("Building feature extractor ...")
    model, preprocess = build_feature_extractor()

    # ------------------------------------------------------------------
    # Part 1: MABO (proposal quality, independent of classifier)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MABO: Mean Average Best Overlap (proposal quality)")
    print("=" * 60)

    mabo, best_overlaps = compute_mabo(proposals_dict, coco_data)
    best_overlaps = np.array(best_overlaps)

    print(f"  MABO           : {mabo:.4f}")
    print(f"  Median BO      : {np.median(best_overlaps):.4f}")
    print(f"  Min BO         : {np.min(best_overlaps):.4f}")
    print(f"  Max BO         : {np.max(best_overlaps):.4f}")
    print(f"  GT boxes       : {len(best_overlaps)}")
    print(f"  BO >= 0.5      : {np.sum(best_overlaps >= 0.5)}/{len(best_overlaps)}")
    print(f"  BO >= 0.75     : {np.sum(best_overlaps >= 0.75)}/{len(best_overlaps)}")

    # ------------------------------------------------------------------
    # Part 2: COCO mAP (end-to-end detection quality)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COCO mAP: Detection quality (proposals + classifier + NMS)")
    print("=" * 60)

    # Build filename -> image_id lookup
    filename_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

    all_detections = []
    total_dets = 0

    for img_info in coco_data['images']:
        filename = img_info['file_name']
        image_id = img_info['id']

        # Load image
        img_path = os.path.join(test_dir, filename)
        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Get proposals
        proposals = proposals_dict.get(filename, [])

        # Detect
        detections = detect_image(
            image, proposals, model, preprocess, svm, scaler,
            conf_threshold=args.conf, nms_threshold=args.nms)

        # Convert to COCO result format
        for det in detections:
            all_detections.append({
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': det['box'],     # [x, y, w, h]
                'score': det['score'],
            })

        total_dets += len(detections)
        print(f"  {filename}: {len(detections)} detections "
              f"(from {len(proposals)} proposals)")

        # Free memory after each image to avoid OOM in WSL2
        del image, proposals, detections
        gc.collect()

    print(f"\n  Total detections: {total_dets}")

    # Run COCO evaluation
    print()
    map_results = evaluate_map(coco_gt, all_detections, cat_id)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  MABO              : {mabo:.4f}")
    if map_results:
        print(f"  AP @[.50:.95]     : {map_results['AP_0.50:0.95']:.4f}")
        print(f"  AP @.50           : {map_results['AP_0.50']:.4f}")
        print(f"  AP @.75           : {map_results['AP_0.75']:.4f}")
        print(f"  AR @[.50:.95]     : {map_results['AR_0.50:0.95']:.4f}")
    print()
    print("  MABO measures proposal quality (how well selective search")
    print("  covers the ground truth objects). A high MABO means the")
    print("  proposals have good spatial coverage of objects.")
    print()
    print("  mAP measures the full pipeline quality (proposals +")
    print("  classification + NMS). It evaluates both localisation")
    print("  precision and detection recall across confidence thresholds.")


if __name__ == '__main__':
    main()