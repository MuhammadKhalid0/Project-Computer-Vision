"""
Task 5.2.2 (feature extraction): Extract CNN features for each training sample.

For each labeled sample (from create_samples.py), we crop the proposal region
from the original image, resize it to 224x224, and pass it through a pre-trained
ResNet18. We use the output of the global average pooling layer (512-dim vector)
as the feature representation for that region.

ResNet18 is a lightweight model that produces discriminative features without
requiring any fine-tuning on our balloon dataset.

Usage:
    python extract_features.py

Input:
    results/samples_train.json
    results/samples_valid.json

Output:
    results/features_train.npz  (features + labels as numpy arrays)
    results/features_valid.npz
"""

import os
import json
import numpy as np
import skimage.io
import torch
import torchvision.models as models
import torchvision.transforms as transforms


# Paths
DATASET_DIR = os.path.join('..', 'data', 'balloon_dataset')
RESULTS_DIR = os.path.join('..', 'results')

# Batch size for CNN inference (adjust if memory is tight)
BATCH_SIZE = 32


def build_feature_extractor():
    """
    Build a ResNet18 feature extractor.

    We load the pre-trained weights and remove the final fully-connected
    classification layer. The model outputs a 512-dimensional feature vector
    per input image (from the global average pooling layer).

    Returns
    -------
    model : torch.nn.Module
        ResNet18 without the classification head, in eval mode.
    preprocess : torchvision.transforms.Compose
        Preprocessing pipeline (resize, normalize) for input crops.
    """
    # Load pre-trained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    resnet = models.resnet18(weights=weights)

    # Remove the final FC layer -> output is 512-dim after avgpool
    # We replace fc with identity so forward() returns the pooled features
    resnet.fc = torch.nn.Identity()
    resnet.eval()

    # Standard ImageNet preprocessing
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
    """
    Crop a bounding box region from an image.

    Parameters
    ----------
    image : np.ndarray
        Original image, shape (H, W, 3).
    box : list
        [x, y, w, h] bounding box (COCO format).

    Returns
    -------
    np.ndarray
        Cropped region, shape (h, w, 3). Clipped to image boundaries.
    """
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    img_h, img_w = image.shape[:2]

    # Clip to image boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    crop = image[y1:y2, x1:x2]

    # Handle degenerate crops (zero width or height)
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        crop = np.zeros((1, 1, 3), dtype=np.uint8)

    return crop


def extract_features_for_split(split_name, model, preprocess):
    """
    Extract CNN features for all labeled samples in a dataset split.

    Parameters
    ----------
    split_name : str
        'train' or 'valid'.
    model : torch.nn.Module
        Feature extractor (ResNet18 without FC).
    preprocess : callable
        Torchvision preprocessing transform.

    Returns
    -------
    features : np.ndarray, shape (N, 512)
        Feature vectors for each sample.
    labels : np.ndarray, shape (N,)
        Class labels (1 = balloon, 0 = background).
    """
    # Load labeled samples
    samples_path = os.path.join(RESULTS_DIR, f'samples_{split_name}.json')
    with open(samples_path, 'r') as f:
        samples = json.load(f)

    split_dir = os.path.join(DATASET_DIR, split_name)
    n_samples = len(samples)
    print(f"\nExtracting features for {split_name} ({n_samples} samples)")

    # Cache loaded images to avoid re-reading the same file
    image_cache = {}

    all_features = []
    all_labels = []
    batch_tensors = []

    for idx, sample in enumerate(samples):
        filename = sample['filename']
        box = sample['box']
        label = sample['label']

        # Load image (with caching)
        if filename not in image_cache:
            img_path = os.path.join(split_dir, filename)
            image_cache[filename] = skimage.io.imread(img_path)

        image = image_cache[filename]

        # Crop the proposal region and preprocess for ResNet
        crop = crop_region(image, box)
        tensor = preprocess(crop)
        batch_tensors.append(tensor)
        all_labels.append(label)

        # Process in batches
        if len(batch_tensors) == BATCH_SIZE or idx == n_samples - 1:
            batch = torch.stack(batch_tensors)
            with torch.no_grad():
                feats = model(batch)  # (batch_size, 512)
            all_features.append(feats.numpy())
            batch_tensors = []

            # Progress update
            done = idx + 1
            print(f"  [{done}/{n_samples}] samples processed")

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels, dtype=np.int32)

    return features, labels


def main():
    print("Building ResNet18 feature extractor ...")
    model, preprocess = build_feature_extractor()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for split_name in ['train', 'valid']:
        features, labels = extract_features_for_split(
            split_name, model, preprocess)

        # Save as compressed numpy archive
        out_path = os.path.join(RESULTS_DIR, f'features_{split_name}.npz')
        np.savez(out_path, features=features, labels=labels)

        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        print(f"  -> Saved {features.shape} features to {out_path}")
        print(f"     ({n_pos} positive, {n_neg} negative)")

    print("\nDone. Features saved to results/ folder.")


if __name__ == '__main__':
    main()

