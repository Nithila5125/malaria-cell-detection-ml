import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2, pandas as pd
import config

df = pd.read_csv(config.RESULTS / "dataset_inventory.csv")

def show(ax, row):
    img = cv2.cvtColor(cv2.imread(str(config.ROOT / row.path)), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"{row.label[:4]} {row.h}x{row.w}", fontsize=7)

fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))
for r, label in enumerate(config.CLASS_MAP):
    s = df[df.label == label].sample(8, random_state=config.SEED)
    for ax, row in zip(axes[r], s.itertuples()):
        show(ax, row)
fig.tight_layout(); fig.savefig(config.RESULTS / "sample_grid.png", dpi=120)

both = df.groupby("name").label.nunique()
names = both[both == 2].index
print("Filenames present in both classes:", len(names))
if len(names):
    pick = pd.Series(names).sample(min(6, len(names)), random_state=config.SEED)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.5), squeeze=False)
    for c, n in enumerate(pick):
        for r, label in enumerate(config.CLASS_MAP):
            show(axes[r, c], df[(df.name == n) & (df.label == label)].iloc[0])
    fig.tight_layout(); fig.savefig(config.RESULTS / "same_name_pairs.png", dpi=120)

t = df.pivot_table(index="slide_code", columns="label", values="name", aggfunc="count", fill_value=0)
t["total"] = t.sum(axis=1)
t.to_csv(config.RESULTS / "slide_summary.csv")
print("Codes only Uninfected:", int(((t.Parasitized == 0) & (t.Uninfected > 0)).sum()),
      "| only Parasitized:", int(((t.Uninfected == 0) & (t.Parasitized > 0)).sum()),
      "| both:", int(((t.Parasitized > 0) & (t.Uninfected > 0)).sum()))
print("Images per code (min/median/max):", int(t.total.min()), int(t.total.median()), int(t.total.max()))
print("Largest code share of all images: %.3f" % (t.total.max() / t.total.sum()))
