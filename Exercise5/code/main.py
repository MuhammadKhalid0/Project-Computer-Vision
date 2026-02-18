'''
@author: Prathmesh R Madhu.
For educational purposes only
'''

# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import os
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from selective_search import selective_search

def main():
    
    # loading a test image from '../data' folder
    image_path = 'data/chrisarch/ca-annun3.jpg'
    image = skimage.io.imread(image_path)
    print (image.shape)
    
    # perform selective search
    image_label, regions = selective_search(
                            image,
                            scale=500,
                            min_size=20
                        )
    
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        
        # excluding regions smaller than 2000 pixels
        # you can experiment using different values for the same
        if r['size'] < 2000:
            continue
        
        # excluding distorted rects
        x, y, w, h = r['rect']
        if w/h > 1.2 or h/w > 1.2:
            continue
        
        candidates.add(r['rect'])
    
    # Draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image)
    for x, y, w, h in candidates:
        print (x, y, w, h, r['size'])
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1
        )
        ax.add_patch(rect)
    plt.axis('off')
    # saving the image
    if not os.path.isdir('results/'):
        os.makedirs('results/')
    fig.savefig('results/'+image_path.split('/')[-1])
    plt.show()


if __name__ == '__main__':
    main()