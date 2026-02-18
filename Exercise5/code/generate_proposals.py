"""
Task 5.2.1: Generate region proposals for the balloon detection dataset.

This script runs selective search on every image in the training and validation
splits of the balloon dataset.
the resulting proposals are saved to JSON files so that subsequent pipeline steps
(sample creation, feature extraction, training) can load them directly.

Usage:
    python generate_proposals.py

Output:
    results/proposals_train.json
    results/proposals_valid.json
    results/proposals_test.json
"""

import os
import sys
import json
import time
import skimage.io

from selective_search import selective_search


#Paths
DATASET_DIR = os.path.join('..', 'data', 'balloon_dataset')
RESULTS_DIR = os.path.join('..', 'results')

#Selective search parameters (changed from ex 5.1)
SS_SCALE = 200
SS_SIGMA = 0.8
SS_MIN_SIZE = 10


def load_coco_annotations(split_dir):
    """
    Load COCO-format annotations for a given dataset split.

    Parameters
    ----------
    split_dir : str
        Path to the split folder (e.g. .../balloon_dataset/train).

    Returns
    -------
    dict
        Parsed JSON content of _annotations.coco.json.
    """
    ann_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(ann_path, 'r') as f:
        coco = json.load(f)
    return coco


def generate_proposals_for_split(split_name):
    """
    Run selective search on every image in a dataset split and collect
    the region proposals.

    Parameters
    ----------
    split_name : str
        One of 'train', 'valid', or 'test'.

    Returns
    -------
    dict
        Dictionary mapping each image filename to a list of proposal
        bounding boxes. Each box is stored as [x, y, w, h] (COCO-style).
    """
    split_dir = os.path.join(DATASET_DIR, split_name)
    coco = load_coco_annotations(split_dir)

    # Build a list of image filenames from the annotations
    image_entries = coco['images']
    num_images = len(image_entries)
    print(f"\nProcessing {split_name} split ({num_images} images)")

    proposals = {}

    for idx, img_info in enumerate(image_entries):
        filename = img_info['file_name']
        img_path = os.path.join(split_dir, filename)

        # Read image
        image = skimage.io.imread(img_path)

        # Some images might be grayscale -> convert to 3 channels
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)

        start = time.time()

        # Run selective search
        _, regions = selective_search(
            image,
            scale=SS_SCALE,
            sigma=SS_SIGMA,
            min_size=SS_MIN_SIZE,
        )

        elapsed = time.time() - start

        # Collect unique bounding boxes as [x, y, w, h]
        boxes = []
        seen = set()
        for r in regions:
            rect = r['rect']  # (x, y, w, h)
            if rect in seen:
                continue
            seen.add(rect)
            boxes.append(list(rect))

        proposals[filename] = boxes

        print(f"  [{idx + 1}/{num_images}] {filename}: "
              f"{len(boxes)} proposals in {elapsed:.1f}s")

    return proposals


def main():
    # Make sure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Process each split
    for split_name in ['train', 'valid', 'test']:
        proposals = generate_proposals_for_split(split_name)

        # Save proposals to a JSON file
        out_path = os.path.join(RESULTS_DIR, f'proposals_{split_name}.json')
        with open(out_path, 'w') as f:
            json.dump(proposals, f)

        total_boxes = sum(len(v) for v in proposals.values())
        print(f"  -> Saved {total_boxes} total proposals "
              f"({len(proposals)} images) to {out_path}")

    print("\nDone. Proposals saved to results/ folder.")


if __name__ == '__main__':
    main()