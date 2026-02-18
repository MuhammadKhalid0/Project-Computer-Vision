'''
Side-by-side comparison of your selective search vs built-in library
'''
# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import skimage.io
import skimage.util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from selective_search import selective_search

def filter_regions(regions):
    """Apply the same filtering as main.py"""
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        
        # excluding distorted rects
        x, y, w, h = r['rect']
        if w/h > 1.2 or h/w > 1.2:
            continue
        
        candidates.add(r['rect'])
    
    return candidates

def draw_regions(ax, image, candidates, title):
    """Draw image with bounding boxes"""
    ax.imshow(image)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1
        )
        ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')

def compare_selective_search():
    """Compare your implementation with built-in library"""
    
    # Load image
    image_path = 'data/classarch/ajax3.jpg'
    image = skimage.io.imread(image_path)
    print(f"Image shape: {image.shape}")
    print("=" * 60)
    
    # Parameters
    scale = 500
    sigma = 0.8
    min_size = 20
    
    # Your implementation
    print("\nRunning My implementation...")
    image_label, regions_yours = selective_search(
        image,
        scale=scale,
        sigma=sigma,
        min_size=min_size
    )
    
    candidates_yours = filter_regions(regions_yours)
    print(f"My implementation: {len(regions_yours)} total regions, {len(candidates_yours)} after filtering")
    
    # Built-in library
    try:
        import selectivesearch
        print("\nRunning BUILT-IN library...")
        
        # Convert to uint8 if needed
        img_uint8 = image if image.dtype == np.uint8 else skimage.util.img_as_ubyte(image)
        
        # Run built-in selective search
        img_lbl, regions_builtin = selectivesearch.selective_search(
            img_uint8,
            scale=scale,
            sigma=sigma,
            min_size=min_size
        )
        
        candidates_builtin = filter_regions(regions_builtin)
        print(f"Built-in library: {len(regions_builtin)} total regions, {len(candidates_builtin)} after filtering")
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
        
        # Your implementation
        draw_regions(ax1, image, candidates_yours, 
                    f'My Implementation\n{len(candidates_yours)} regions')
        
        # Built-in library
        draw_regions(ax2, image, candidates_builtin,
                    f'Built-in Library\n{len(candidates_builtin)} regions')
        
        plt.tight_layout()
        
        if not os.path.isdir('results/'):
            os.makedirs('results/')
        fig.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to results/comparison.png")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("COMPARISON STATISTICS:")
        print("=" * 60)
        print(f"Total regions:")
        print(f"  Your implementation: {len(regions_yours)}")
        print(f"  Built-in library:    {len(regions_builtin)}")
        print(f"  Difference:         {len(regions_builtin) - len(regions_yours)}")
        print(f"\nAfter filtering:")
        print(f"  Your implementation: {len(candidates_yours)}")
        print(f"  Built-in library:    {len(candidates_builtin)}")
        print(f"  Difference:          {len(candidates_builtin) - len(candidates_yours)}")
        
        # Show some example regions
        print(f"\nFirst 5 regions from your implementation:")
        for i, (x, y, w, h) in enumerate(list(candidates_yours)[:5]):
            print(f"  {i+1}. rect=({x}, {y}, {w}, {h})")
        
        print(f"\nFirst 5 regions from built-in library:")
        for i, (x, y, w, h) in enumerate(list(candidates_builtin)[:5]):
            print(f"  {i+1}. rect=({x}, {y}, {w}, {h})")
        
        plt.show()
        
    except ImportError:
        print("\nERROR: 'selectivesearch' library not installed!")
        print("To install: pip install selectivesearch")
        print("\nShowing only your implementation:")
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        draw_regions(ax, image, candidates_yours,
                    f'My Implementation\n{len(candidates_yours)} regions')
        
        if not os.path.isdir('results/'):
            os.makedirs('results/')
        fig.savefig('results/your_implementation.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    compare_selective_search()