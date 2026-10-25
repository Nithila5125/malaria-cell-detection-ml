import sys, time
import cv2, numpy as np, pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import config
from src.masking import cell_mask, morphology

PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]

def stats(prefix, v):
    q = np.percentile(v, [5, 50, 95])
    return {f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(),
            f"{prefix}_p5": q[0], f"{prefix}_p50": q[1], f"{prefix}_p95": q[2]}

def features(path):
    bgr = cv2.imread(str(config.ROOT / path))
    m = cell_mask(bgr)
    f = morphology(m)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV_FULL).astype(float)
    for i, c in enumerate("RGB"):
        f.update(stats(c, rgb[..., i][m].astype(float)))
    hue = hsv[..., 0][m] * 2 * np.pi / 255
    f["H_sin_mean"], f["H_cos_mean"] = np.sin(hue).mean(), np.cos(hue).mean()
    f.update(stats("S", hsv[..., 1][m]))
    f.update(stats("V", hsv[..., 2][m]))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    q = np.where(m, np.minimum(gray // 8, 31) + 1, 0).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=33, symmetric=True)
    g = g[1:, 1:, :, :].astype(float)
    g /= g.sum(axis=(0, 1), keepdims=True)
    for p in PROPS:
        f[f"glcm_{p}"] = graycoprops(g, p).mean() if p != "ASM" else (g ** 2).sum(axis=(0, 1)).mean()
    return f

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sp = pd.read_csv(config.PROCESSED / "split.csv")
    if n:
        sp = sp.groupby("label").sample(n // 2, random_state=config.SEED)
    t = time.time(); rows = []
    for r in tqdm(sp.itertuples(), total=len(sp)):
        rows.append(features(r.path))
    X = pd.DataFrame(rows, index=sp.index)
    out = pd.concat([sp[["path", "name", "label", "y", "slide_code", "split"]], X], axis=1)
    name = "features_smoke.parquet" if n else "features.parquet"
    out.to_parquet(config.PROCESSED / name, index=False)
    print("rows", len(out), "| features", X.shape[1], "| NaN cells", int(X.isna().sum().sum()),
          "| inf cells", int(np.isinf(X.to_numpy()).sum()), "| seconds %.1f" % (time.time() - t))
