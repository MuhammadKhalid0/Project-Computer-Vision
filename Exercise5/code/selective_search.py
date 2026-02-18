'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    
    The Felzenszwalb algorithm performs efficient graph-based image segmentation
    by treating pixels as graph nodes and connecting neighboring pixels with edges
    weighted by their dissimilarity. The algorithm adaptively adjusts segmentation
    criteria based on local image variability, allowing it to preserve detail in
    low-variability regions while ignoring detail in high-variability regions.
    
    Parameters:
    -----------
    im_orig : numpy.ndarray
        Input image with shape (height, width, 3) in RGB format
    scale : float
        Controls the cluster size - higher values result in larger segments.
        This parameter influences the threshold for merging regions.
    sigma : float
        Width of the Gaussian kernel used for smoothing the image before
        computing the segmentation. Helps reduce noise sensitivity.
    min_size : int
        Minimum component size in pixels. Regions smaller than this will be
        merged with neighboring regions.
    
    Returns:
    --------
    numpy.ndarray
        Image with segmentation labels added as a 4th channel.
        Shape: (height, width, 4) where the 4th channel contains region labels.
    """
    segmentation_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig),
        scale=scale,
        sigma=sigma,
        min_size=min_size,
    )   
    
    # Merge mask channel to the image as a 4th channel
    # np.append upcasts uint8 to float64, keeping RGB values as 0-255 float
    # This avoids uint8 overflow when there are more than 256 segments
    im_orig = np.append(
        im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = segmentation_mask
    
    return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    
    The color similarity between two regions is computed using histogram intersection,
    which measures how similar the color distributions are. The formula sums the minimum
    values of corresponding histogram bins from both regions.
    
    Formula: s_colour(r_i, r_j) = Σ_{k=1}^n min(c_i^k, c_j^k)
    
    Parameters:
    -----------
    r1 : dict
        First region dictionary containing 'colour_hist' key with color histogram
    r2 : dict
        Second region dictionary containing 'colour_hist' key with color histogram
    
    Returns:
    --------
    float
        Color similarity score between 0 and 1, where 1 indicates identical
        color distributions and 0 indicates completely different distributions.
    """
    # Extract color histograms from both regions
    hist1 = r1["colour_hist"]
    hist2 = r2["colour_hist"]
    
    # Calculate histogram intersection: sum of minimum values for each bin
    # This measures the overlap between the two color distributions
    intersection = np.sum(np.minimum(hist1, hist2))
    
    return intersection


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    
    Texture similarity is computed using histogram intersection on texture histograms,
    similar to color similarity. The texture histograms capture local binary patterns
    (LBP) or gradient information across different color channels.
    
    Formula: S_texture(r_i, r_j) = Σ_{k=1}^{n} min(t_i^k, t_j^k)
    
    Parameters:
    -----------
    r1 : dict
        First region dictionary containing 'texture_hist' key with texture histogram
    r2 : dict
        Second region dictionary containing 'texture_hist' key with texture histogram
    
    Returns:
    --------
    float
        Texture similarity score between 0 and 1, where 1 indicates identical
        texture distributions and 0 indicates completely different distributions.
    """
    # Extract texture histograms from both regions
    hist1 = r1["texture_hist"]
    hist2 = r2["texture_hist"]
    
    # Calculate histogram intersection: sum of minimum values for each bin
    # This measures the overlap between the two texture distributions
    intersection = np.sum(np.minimum(hist1, hist2))
    
    return intersection


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    
    Size similarity encourages merging of smaller regions into larger ones. The
    similarity is higher when the combined size of two regions is small relative
    to the total image size, promoting the merging of small regions.
    
    Formula: s_size(r_i, r_j) = 1 - (size(r_i) + size(r_j)) / size(im)
    
    Parameters:
    -----------
    r1 : dict
        First region dictionary containing 'size' key with number of pixels
    r2 : dict
        Second region dictionary containing 'size' key with number of pixels
    imsize : int
        Total number of pixels in the image (height * width)
    
    Returns:
    --------
    float
        Size similarity score between 0 and 1, where values closer to 1 indicate
        smaller regions that should be merged, and values closer to 0 indicate
        larger regions that are less likely to merge.
    """
    # Calculate the combined size of both regions
    combined_size = r1["size"] + r2["size"]
    
    # Size similarity: higher when regions are small relative to image
    # This encourages merging of small regions into larger segmentations
    similarity = 1.0 - (combined_size / float(imsize))
    
    return similarity


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    
    Fill similarity measures how well two regions fit together by comparing the
    area of their combined bounding box to the sum of their individual sizes.
    Regions that fit well together (low gap in bounding box) have higher similarity.
    
    Formula: fill(r_i, r_j) = 1 - (size(BB_ij) - size(r_i) - size(r_j)) / size(im)
    
    Where BB_ij is the bounding box that tightly encloses both regions r_i and r_j.
    
    Parameters:
    -----------
    r1 : dict
        First region dictionary containing 'min_x', 'max_x', 'min_y', 'max_y', 'size'
    r2 : dict
        Second region dictionary containing 'min_x', 'max_x', 'min_y', 'max_y', 'size'
    imsize : int
        Total number of pixels in the image (height * width)
    
    Returns:
    --------
    float
        Fill similarity score between 0 and 1, where values closer to 1 indicate
        regions that fit well together (small gap in bounding box), and values
        closer to 0 indicate regions with large gaps between them.
    """
    # Calculate the bounding box that encloses both regions
    # The bounding box spans from the minimum to maximum coordinates
    bb_min_x = min(r1["min_x"], r2["min_x"])
    bb_max_x = max(r1["max_x"], r2["max_x"])
    bb_min_y = min(r1["min_y"], r2["min_y"])
    bb_max_y = max(r1["max_y"], r2["max_y"])
    
    # Calculate the area of the bounding box
    bb_width = bb_max_x - bb_min_x
    bb_height = bb_max_y - bb_min_y
    bb_size = bb_width * bb_height
        
    # Calculate the gap: area of bounding box minus the actual region sizes
    # This gap represents empty space between or around the regions
    gap = bb_size - r1["size"] - r2["size"]
    
    # Fill similarity: higher when gap is small relative to image size
    # Regions that fit well together have small gaps
    similarity = 1.0 - (gap / float(imsize))
    
    return similarity

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings proposal in the paper]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    
    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]
        
        # calculate histogram for each colour and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])
    
    # L1 normalize
    hist = hist / len(img)
    
    return hist

def calc_texture_gradient(img):
    """
    Task 2.5.2
    Parameters:
    -----------
    img : numpy.ndarray
        Input image with shape (height, width, 3) in RGB format
    
    Returns:
    --------
    numpy.ndarray
        Texture gradient image with shape (height, width, 3), where each channel
        contains LBP values for the corresponding RGB channel.
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    
    # Compute LBP for each color channel separately
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)
    
    return ret

def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    
    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])
    
    # L1 Normalize
    hist = hist / len(img)
    
    return hist

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    
    # Extract RGB image (first 3 channels) and segment labels (4th channel)
    rgb_img = img[:, :, :3]
    segment_labels = img[:, :, 3]
    
    # get hsv image
    hsv = skimage.color.rgb2hsv(rgb_img)
    
    # pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
            
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
    
    # pass 2: calculate texture gradient
    tex_grad = calc_texture_gradient(img)
    
    # pass 3: calculate colour histogram of each region
    for k, v in R.items():
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels)
        R[k]["colour_hist"] = calc_colour_hist(masked_pixels)
        
        # texture histogram
        R[k]["texture_hist"] = calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
    
    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    
    # Get all region IDs
    region_ids = list(regions.keys())
    
    # Check all pairs of regions for intersection
    for i in range(len(region_ids)):
        for j in range(i + 1, len(region_ids)):
            region_id_a = region_ids[i]
            region_id_b = region_ids[j]
            region_a = regions[region_id_a]
            region_b = regions[region_id_b]
            
            # Check if bounding boxes intersect (neighbours)
            if intersect(region_a, region_b):
                # Add as tuple: ((region_id_a, region_dict_a), (region_id_b, region_dict_b))
                neighbours.append(((region_id_a, region_a), (region_id_b, region_b)))

    return neighbours

def merge_regions(r1, r2):
    """
    Merge two regions into a single region.
    Combines bounding boxes, sizes, and histograms (weighted by size).
    The weighted average preserves the L1 norm of the histograms,
    so no re-normalization is needed.
    """
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "colour_hist": (r1["size"] * r1["colour_hist"]
                        + r2["size"] * r2["colour_hist"]) / new_size,
        "texture_hist": (r1["size"] * r1["texture_hist"]
                         + r2["size"] * r2["texture_hist"]) / new_size,
    }
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, []

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Add labels to initial regions to track which segments they contain
    for region_id in R:
        R[region_id]["labels"] = [region_id]

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(min(ai, bi), max(ai, bi))] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda x: x[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])
        # Combine labels from both merged regions
        R[t]["labels"] = R[i]["labels"] + R[j]["labels"]

        # Task 5: Mark similarities for regions to be removed
        keys_to_remove = []
        neighbour_ids = set()
        for key in list(S.keys()):
            if key[0] == i or key[0] == j or key[1] == i or key[1] == j:
                keys_to_remove.append(key)
                # Collect the "other" region in each pair
                if key[0] == i or key[0] == j:
                    neighbour_ids.add(key[1])
                if key[1] == i or key[1] == j:
                    neighbour_ids.add(key[0])

        # Remove i and j from the neighbor set (they are being merged)
        neighbour_ids.discard(i)
        neighbour_ids.discard(j)

        # Task 6: Remove old similarities of related regions
        for key in keys_to_remove:
            del S[key]

        # Task 7: Calculate similarities with the new region
        # Compute similarity between t and each of its neighbors
        for k in neighbour_ids:
            S[(min(t, k), max(t, k))] = calc_sim(R[t], R[k], imsize)

    # Task 8: Generating the final regions from R
    # R now contains ALL regions: initial segments + all merged regions
    regions = []
    for region_id, region in R.items():
        # Calculate bounding box: (left, top, width, height)
        left = region["min_x"]
        top = region["min_y"]
        width = region["max_x"] - region["min_x"]
        height = region["max_y"] - region["min_y"]
        
        # Create region dictionary
        region_dict = {
            'rect': (left, top, width, height),
            'labels': region.get("labels", [region_id]),
            'size': region["size"]
        }
        regions.append(region_dict)

    return image, regions