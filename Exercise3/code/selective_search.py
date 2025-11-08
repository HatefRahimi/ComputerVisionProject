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

from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv
from skimage.feature import local_binary_pattern
from itertools import combinations
from typing import Dict, List, Tuple


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    # 1.1: run Felzenszwalb over-segmentation
    segments = felzenszwalb(im_orig,
                            scale=scale,
                            sigma=sigma,
                            min_size=min_size)

    # 1.2: append the segment IDs as a 4th channel
    #     (convert to float, normalize if you like)
    seg_channel = segments.astype(np.float32)[..., np.newaxis]
    out_image = np.concatenate([im_orig, seg_channel], axis=2)

    return out_image


def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###
    h1 = r1['colour_hist']
    h2 = r2['colour_hist']
    # intersection: sum of minimum bin values
    return float(np.minimum(h1, h2).sum())


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###
    t1 = r1['texture_hist']
    t2 = r2['texture_hist']
    return float(np.minimum(t1, t2).sum())


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###
    s1 = r1['size']
    s2 = r2['size']
    return 1.0 - float(s1 + s2) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###
    # union bounding box
    x1a, y1a, x2a, y2a = r1['bbox']
    x1b, y1b, x2b, y2b = r2['bbox']
    ux1, uy1 = min(x1a, x1b), min(y1a, y1b)
    ux2, uy2 = max(x2a, x2b), max(y2a, y2b)
    union_area = float((ux2 - ux1) * (uy2 - uy1))

    # subtract the two region areas (they may overlap, but fill penalizes extra bg)
    extra = union_area - (r1['size'] + r2['size'])
    return 1.0 - (extra / imsize)


def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))


def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    ### YOUR CODE HERE ###
    hsv = rgb2hsv(img)
    # 25 bins per channel → 75-dim vector
    histograms = [
        np.histogram(hsv[..., ch].ravel(), bins=BINS, range=(0, 1))[0].astype(float)
        for ch in range(3)
    ]
    hist = np.concatenate(histograms)

    return hist / (hist.sum() + 1e-6)


def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    # ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    radius = 1
    n_points = 8 * radius
    # stack LBP from each channel along the last axis
    lbp_channels = []
    for ch in range(img.shape[-1]):
        channel = img[..., ch]
        # if float, scale to [0,255] and convert
        if np.issubdtype(channel.dtype, np.floating):
            channel = (channel * 255).astype(np.uint8)
        lbp = local_binary_pattern(
            channel,
            P=n_points,
            R=radius,
            method='uniform'
        )
        lbp_channels.append(lbp)
    return np.stack(lbp_channels, axis=-1)


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    # hist = np.array([])
    ### YOUR CODE HERE ###
    C = img.shape[-1]
    histograms = []
    for ch in range(C):
        data = img[..., ch].ravel()
        if data.size == 0:
            # empty region → zero histogram
            h = np.zeros(BINS)
        else:
            # avoid zero-length range if max==0
            max_val = data.max()
            range_max = max(max_val, 1)
            h, _ = np.histogram(data, bins=BINS, range=(0, range_max))
        histograms.append(h.astype(float))
    hist = np.concatenate(histograms)
    return hist / (hist.sum() + 1e-6)


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
    ### YOUR CODE HERE ###
    H, W = img.shape[:2]
    segmap = img[..., 3].astype(int)
    hsv_img = rgb2hsv(img[..., :3])
    tex_grad = calc_texture_gradient(img[..., :3])

    for label in np.unique(segmap):
        mask = (segmap == label)
        ys, xs = np.nonzero(mask)
        size = int(mask.sum())
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # gather pixel-level data
        region_hsv = hsv_img[mask]
        region_tex = tex_grad[mask]

        R[label] = {
            'labels': (ys * W + xs).tolist(),
            'size': size,
            'bbox': bbox,
            'colour_hist': calc_colour_hist(region_hsv),
            'texture_hist': calc_texture_hist(region_tex),
        }

    return R


# def intersect(a, b):
#     if (a["min_x"] < b["min_x"] < a["max_x"]
#         and a["min_y"] < b["min_y"] < a["max_y"]) or (
#             a["min_x"] < b["max_x"] < a["max_x"]
#             and a["min_y"] < b["max_y"] < a["max_y"]) or (
#             a["min_x"] < b["min_x"] < a["max_x"]
#             and a["min_y"] < b["max_y"] < a["max_y"]) or (
#             a["min_x"] < b["max_x"] < a["max_x"]
#             and a["min_y"] < b["min_y"] < a["max_y"]):
#         return True
#     return False

def intersect(a: dict, b: dict) -> bool:
    """
    Return True if the bounding-boxes of regions a and b overlap.
    Uses axis-aligned bounding boxes stored in a['bbox'] and b['bbox'].
    """
    ax1, ay1, ax2, ay2 = a['bbox']
    bx1, by1, bx2, by2 = b['bbox']
    # check for separation on x or y axis
    if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
        return False
    return True


def extract_neighbours(regions):
    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###

    return [
        (i, j)
        for i, j in combinations(regions.keys(), 2)
        if intersect(regions[i], regions[j])
    ]


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
    size1, size2 = r1['size'], r2['size']
    total = size1 + size2

    # weighted average of histograms
    colour = (r1['colour_hist'] * size1 + r2['colour_hist'] * size2) / total
    texture = (r1['texture_hist'] * size1 + r2['texture_hist'] * size2) / total

    # merged bounding box
    x1 = min(r1['bbox'][0], r2['bbox'][0])
    y1 = min(r1['bbox'][1], r2['bbox'][1])
    x2 = max(r1['bbox'][2], r2['bbox'][2])
    y2 = max(r1['bbox'][3], r2['bbox'][3])

    return {
        'labels': r1['labels'] + r2['labels'],
        'size': total,
        'bbox': (x1, y1, x2, y2),
        'colour_hist': colour,
        'texture_hist': texture
    }


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
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    max_regions = 2000

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    # neighbours = extract_neighbours(R)
    #
    # # Calculating initial similarities
    # S = {}
    # # for (ai, ar), (bi, br) in neighbours:
    # #     S[(ai, bi)] = calc_sim(ar, br, imsize)
    # for i, j in neighbours:
    #     ri, rj = R[i], R[j]  # look up the region dicts
    #     S[(i, j)] = calc_sim(ri, rj, imsize)
    #
    # # Hierarchical search for merging similar regions
    # while S != {}:
    #
    #     # Get highest similarity
    #     i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
    #
    #     # Task 4: Merge corresponding regions. Refer to function "merge_regions"
    #     t = max(R.keys()) + 1.0
    #     R[t] = merge_regions(R[i], R[j])
    #
    #     # Task 5: Mark similarities for regions to be removed
    #     ### YOUR CODE HERE ###
    #     to_remove = [pair for pair in S if i in pair or j in pair]
    #
    #     # Task 6: Remove old similarities of related regions
    #     ### YOUR CODE HERE ###
    #     for pair in to_remove:
    #         del S[pair]
    #
    #     # Remove the old regions themselves so they’re not reused
    #     del R[i], R[j]
    #
    #     # Task 7: Calculate similarities with the new region
    #     ### YOUR CODE HERE ###
    #     for k in R:
    #         if k == t:
    #             continue
    #         if intersect(R[t], R[k]):
    #             S[(t, k)] = calc_sim(R[t], R[k], imsize)
    #
    # # Task 8: Generating the final regions from R
    # regions = []
    # ### YOUR CODE HERE ###
    # for reg in R.values():
    #     x1, y1, x2, y2 = reg['bbox']
    #     w, h = x2 - x1, y2 - y1
    #
    #     regions.append({
    #         'rect': (x1, y1, w, h),
    #         'labels': reg['labels'],
    #         'size': reg['size']
    #     })
    #
    # return image, regions

    # 3) Record all initial superpixel boxes
    proposals = []
    for reg in R.values():
        x1, y1, x2, y2 = reg['bbox']
        proposals.append({
            'rect': (x1, y1, x2 - x1, y2 - y1),
            'labels': reg['labels'],
            'size': reg['size']
        })

    # 4) Build initial neighbour list & similarity map
    neighbours = extract_neighbours(R)  # returns list of (i, j)
    S = {}
    for i, j in neighbours:
        S[(i, j)] = calc_sim(R[i], R[j], imsize)

    # 5) Hierarchical merging
    while S and len(R) < max_regions:
        # pick highest-similarity pair
        (i, j), _ = max(S.items(), key=lambda kv: kv[1])

        # merge regions i and j into new region t
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # remove old similarities involving i or j
        to_remove = [pair for pair in S if i in pair or j in pair]
        for pair in to_remove:
            del S[pair]

        # drop the merged regions
        del R[i], R[j]

        # compute similarities between t and all other regions
        for k in R:
            if k == t:
                continue
            if intersect(R[t], R[k]):
                S[(t, k)] = calc_sim(R[t], R[k], imsize)

        # record this new merged region as a proposal
        xr1, yr1, xr2, yr2 = R[t]['bbox']
        proposals.append({
            'rect': (xr1, yr1, xr2 - xr1, yr2 - yr1),
            'labels': R[t]['labels'],
            'size': R[t]['size']
        })

    # 6) Return the 4-channel image and full list of proposals
    return image, proposals
