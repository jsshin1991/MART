import numpy as np
from tqdm import tqdm
import os
from skimage.segmentation import slic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def min(x, y):
    if x < y:
        return x
    return y


def weight_function(x):
    return min(1 / (np.sum(np.abs(x) + 1e-12)), 1e12)


def neigh_search(img_seg, partial_segment, idx):
    basis = img_seg[idx[0], idx[1]]
    if (idx[0] + 1, idx[1]) not in partial_segment and idx[0] + 1 < 224 and img_seg[idx[0] + 1, idx[1]] == basis:
        partial_segment.append((idx[0] + 1, idx[1]))
        neigh_search(img_seg, partial_segment, (idx[0] + 1, idx[1]))
    if (idx[0], idx[1] + 1) not in partial_segment and idx[1] + 1 < 224 and img_seg[idx[0], idx[1] + 1] == basis:
        partial_segment.append((idx[0], idx[1] + 1))
        neigh_search(img_seg, partial_segment, (idx[0], idx[1] + 1))


def run_superpixel_MART(input, target_label_index, predictions, iter=5, n_segments=1000):
    imageSegmentNDArray = slic(input, compactness=10, n_segments=n_segments)
    candidate = [(i, j) for i in range(224) for j in range(224)]
    segment = []
    for i in range(224):
        for j in range(224):
            if (i, j) in candidate:
                partial_segment = [(i, j)]
                neigh_search(imageSegmentNDArray, partial_segment, (i, j))
                candidate = list(set(candidate) - set(partial_segment))
                segment.append(list(set(partial_segment)))

    c_i = np.zeros_like(input, dtype=np.float)
    mask = np.zeros_like(input, dtype=np.float)
    x = input.copy().astype('float')
    input_pred = predictions([input], target_label_index)

    min = 0
    max = 255

    for total_it in tqdm(range(len(segment))):
        obj_pixels = segment[total_it]
        for idx in obj_pixels:
            mask[idx[0], idx[1], :] = 1
        e = [min - input[obj_pixels[0][0], obj_pixels[0][1], k] for k in range(3)]
        for it in range(iter + 1):
            for k in range(3):
                x += e[k] * mask
                w = weight_function(x - input)
                diff = np.abs(predictions([x], target_label_index) - input_pred)[:, target_label_index]
                e[k] = (max - min) / iter
                for idx in obj_pixels:
                    c_i[idx[0], idx[1], k] += w * diff
        for idx in obj_pixels:
            x[idx[0], idx[1], :] = input[idx[0], idx[1], :]
        mask.fill(0)

    mart = c_i / c_i.sum()
    return mart
