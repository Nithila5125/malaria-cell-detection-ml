import cv2, numpy as np
from scipy.ndimage import binary_fill_holes
from skimage import measure

def cell_mask(bgr, thr=10):
    m = bgr.max(axis=2) > thr
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    k = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
    return binary_fill_holes(lab == k)

def morphology(mask):
    r = measure.regionprops(mask.astype(np.uint8))[0]
    p = r.perimeter
    return dict(area=float(r.area), perimeter=float(p),
                circularity=float(4 * np.pi * r.area / p**2) if p > 0 else np.nan,
                eccentricity=float(r.eccentricity), solidity=float(r.solidity))
