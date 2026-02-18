"""
Task 5.2.4: Inference script for balloon detection.

Full detection pipeline on an arbitrary input image:
    1. Generate region proposals using selective search (from Ex 5.1)
    2. Extract CNN features for each proposal (ResNet18, same as training)
    3. Classify proposals with the trained SVM
    4. Apply Non-Maximum Suppression (NMS) to remove duplicate detections
    5. Visualise the final detections on the image

Usage:
    python detect.py <image_path> [--conf 0.5] [--nms 0.3]

Example:
    python detect.py ../data/balloon_dataset/test/some_image.jpg
    python detect.py ../data/balloon_dataset/test/some_image.jpg --conf 0.3 --nms 0.4
"""

import os
import gc
import sys
import argparse
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import joblib

from selective_search import selective_search


# Paths
RESULTS_DIR = os.path.join('..', 'results')

# Selective search parameters (same as training)
SS_SCALE = 200
SS_SIGMA = 0.8
SS_MIN_SIZE = 10


# ------------------------------------------------------------------
# Feature extraction (same setup as extract_features.py)
# ------------------------------------------------------------------

def build_feature_extractor():
    """Load pre-trained ResNet18 as a 512-dim feature extractor."""
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return resnet, preprocess


def crop_region(image, box):
    """Crop a [x, y, w, h] region from the image, clipped to boundaries."""
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img_h, img_w = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    crop = image[y1:y2, x1:x2]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


# ------------------------------------------------------------------
# Non-Maximum Suppression
# ------------------------------------------------------------------

def non_maximum_suppression(boxes, scores, iou_threshold=0.3):
    """
    Greedy Non-Maximum Suppression to remove overlapping detections.

    For each pair of overlapping boxes, only the one with higher confidence
    is kept. This prevents multiple detections of the same object.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4)
        Bounding boxes as [x, y, w, h].
    scores : np.ndarray, shape (N,)
        Confidence scores for each box.
    iou_threshold : float
        Boxes with IoU above this threshold are suppressed.

    Returns
    -------
    list of int
        Indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    # Sort by confidence (ascending), process from highest
    order = scores.argsort()

    keep = []
    while order.size > 0:
        # Pick the box with highest score
        i = order[-1]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU of this box with all remaining boxes
        remaining = order[:-1]
        xx1 = np.maximum(x1[i], x1[remaining])
        yy1 = np.maximum(y1[i], y1[remaining])
        xx2 = np.minimum(x2[i], x2[remaining])
        yy2 = np.minimum(y2[i], y2[remaining])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = areas[i] + areas[remaining] - inter_area
        iou = inter_area / np.maximum(union_area, 1e-6)

        # Keep only boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = remaining[mask]

    return keep


# ------------------------------------------------------------------
# Detection pipeline
# ------------------------------------------------------------------

def detect(image, model, preprocess, svm, scaler,
           conf_threshold=0.5, nms_threshold=0.3):
    """
    Run the full detection pipeline on a single image.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image, shape (H, W, 3).
    model : torch.nn.Module
        ResNet18 feature extractor.
    preprocess : callable
        Torchvision preprocessing transform.
    svm : sklearn SVC
        Trained SVM classifier.
    scaler : sklearn StandardScaler
        Fitted feature scaler.
    conf_threshold : float
        Minimum confidence to keep a detection.
    nms_threshold : float
        IoU threshold for NMS.

    Returns
    -------
    detections : list of dict
        Each detection: {'box': [x,y,w,h], 'score': float, 'label': str}
    n_proposals : int
        Total number of proposals generated.
    """
    # Step 1: Generate region proposals
    print("  Generating region proposals ...")
    _, regions = selective_search(
        image, scale=SS_SCALE, sigma=SS_SIGMA, min_size=SS_MIN_SIZE)

    # Deduplicate proposals
    boxes = []
    seen = set()
    for r in regions:
        rect = r['rect']
        if rect in seen:
            continue
        seen.add(rect)
        boxes.append(list(rect))

    n_proposals = len(boxes)
    print(f"  {n_proposals} unique proposals")

    if n_proposals == 0:
        return [], 0

    # Step 2: Extract features in mini-batches to avoid OOM
    print("  Extracting features ...")
    BATCH_SIZE = 64
    all_features = []
    for start in range(0, len(boxes), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(boxes))
        batch_tensors = []
        for box in boxes[start:end]:
            crop = crop_region(image, box)
            tensor = preprocess(crop)
            batch_tensors.append(tensor)
        batch = torch.stack(batch_tensors)
        with torch.no_grad():
            feats = model(batch).numpy()
        all_features.append(feats)
        del batch, batch_tensors
    features = np.concatenate(all_features, axis=0)  # (N, 512)
    del all_features
    gc.collect()

    # Step 3: Standardise and classify with SVM
    features = scaler.transform(features)
    predictions = svm.predict(features)
    probabilities = svm.predict_proba(features)  # (N, 2)
    # Column 1 = probability of class "balloon"
    balloon_scores = probabilities[:, 1]

    # Step 4: Filter by confidence threshold and class
    det_boxes = []
    det_scores = []
    for i, (pred, score) in enumerate(zip(predictions, balloon_scores)):
        if pred == 1 and score >= conf_threshold:
            det_boxes.append(boxes[i])
            det_scores.append(score)

    print(f"  {len(det_boxes)} detections above confidence {conf_threshold}")

    if len(det_boxes) == 0:
        return [], n_proposals

    det_boxes = np.array(det_boxes, dtype=np.float32)
    det_scores = np.array(det_scores)

    # Step 5: Non-Maximum Suppression
    keep = non_maximum_suppression(det_boxes, det_scores, nms_threshold)
    print(f"  {len(keep)} detections after NMS (IoU={nms_threshold})")

    detections = []
    for i in keep:
        detections.append({
            'box': det_boxes[i].tolist(),
            'score': float(det_scores[i]),
            'label': 'balloon',
        })

    return detections, n_proposals


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def visualise_detections(image, detections, title="", save_path=None):
    """
    Draw detected bounding boxes on the image.

    Parameters
    ----------
    image : np.ndarray
        Original RGB image.
    detections : list of dict
        Each with 'box', 'score', 'label'.
    title : str
        Plot title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for det in detections:
        x, y, w, h = det['box']
        score = det['score']

        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        ax.text(x, y - 4,
                f"balloon {score:.2f}",
                color='white', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.7, pad=1))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect balloons in an input image.")
    parser.add_argument('image_path', type=str,
                        help='Path to the input image')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--nms', type=float, default=0.3,
                        help='NMS IoU threshold (default: 0.3)')
    args = parser.parse_args()

    # Load image
    print(f"Loading image: {args.image_path}")
    image = skimage.io.imread(args.image_path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Load trained model and scaler
    print("Loading SVM model and scaler ...")
    svm = joblib.load(os.path.join(RESULTS_DIR, 'svm_model.joblib'))
    scaler = joblib.load(os.path.join(RESULTS_DIR, 'svm_scaler.joblib'))

    # Build feature extractor
    print("Building feature extractor ...")
    model, preprocess = build_feature_extractor()

    # Run detection
    print("Running detection pipeline ...")
    detections, n_proposals = detect(
        image, model, preprocess, svm, scaler,
        conf_threshold=args.conf, nms_threshold=args.nms)

    # Print results
    print(f"\nResults: {len(detections)} balloon(s) detected "
          f"(from {n_proposals} proposals)")
    for i, det in enumerate(detections):
        x, y, w, h = det['box']
        print(f"  [{i+1}] box=({x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}), "
              f"score={det['score']:.3f}")

    # Visualise
    img_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(RESULTS_DIR, f'detection_{img_name}.png')
    visualise_detections(
        image, detections,
        title=f"Detected {len(detections)} balloon(s)",
        save_path=save_path)


if __name__ == '__main__':
    main()

