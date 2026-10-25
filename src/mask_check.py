import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2, pandas as pd
import config
from src.masking import cell_mask, morphology

df = pd.read_csv(config.RESULTS / "dataset_inventory.csv")
df["crop_area"] = df.h * df.w
picks = []
for label in config.CLASS_MAP:
    d = df[df.label == label].sort_values("crop_area")
    picks += [d.head(2), d.tail(2), d.sample(4, random_state=config.SEED)]
sel = pd.concat(picks)

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for ax, row in zip(axes.ravel(), sel.itertuples()):
    bgr = cv2.imread(str(config.ROOT / row.path))
    m = cell_mask(bgr)
    mo = morphology(m)
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.contour(m.astype(float), levels=[0.5], colors="lime", linewidths=1)
    ax.set_title(f"{row.label[:4]} A={mo['area']:.0f} C={mo['circularity']:.2f}", fontsize=7)
    ax.axis("off")
fig.tight_layout(); fig.savefig(config.RESULTS / "mask_check.png", dpi=120)

rs = df.groupby("label").sample(500, random_state=config.SEED)
rows = []
for row in rs.itertuples():
    bgr = cv2.imread(str(config.ROOT / row.path))
    m = cell_mask(bgr)
    raw = int((bgr.max(axis=2) > 10).sum())
    rows.append(dict(label=row.label, area=int(m.sum()),
                     frac=m.sum() / (row.h * row.w), ratio=m.sum() / raw if raw else 0))
s = pd.DataFrame(rows)
print("Empty masks:", int((s.area == 0).sum()))
print(s.groupby("label").frac.describe()[["min", "50%", "max"]])
print("Stray pixels dropped (ratio<0.99):", int((s.ratio < 0.99).sum()),
      "| holes filled (ratio>1.01):", int((s.ratio > 1.01).sum()))
