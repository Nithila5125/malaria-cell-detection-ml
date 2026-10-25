import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

name, mode = "lr", "all"
p = pd.read_csv(config.RESULTS / f"test_predictions_{mode}.csv")
f = pd.read_parquet(config.PROCESSED / "features.parquet",
                    columns=["path", "area", "S_p95", "S_std", "glcm_contrast", "V_p5"])
d = p.merge(f, on="path")
d["err"] = d[f"{name}_pred"] != d.y
d["kind"] = np.select([(d.y == 1) & d.err, (d.y == 0) & d.err, d.y == 1, d.y == 0], ["FN", "FP", "TP", "TN"], default="")
print(d.kind.value_counts().to_string())

s = d.groupby("slide_code").agg(n=("err", "size"), errors=("err", "sum"), parasitized=("y", "sum"))
s["err_rate"] = (s.errors / s.n).round(3)
s = s.sort_values("errors", ascending=False)
s.to_csv(config.RESULTS / f"error_by_slide_{name}.csv")
print(s.head(8).to_string())
print("Share of all errors in the 5 worst slides: %.2f" % (s.errors.head(5).sum() / s.errors.sum()))
print("Slides with zero errors:", int((s.errors == 0).sum()), "of", len(s))
print(d.groupby("kind")[["area", "S_p95", "S_std", "glcm_contrast", "V_p5"]].median().round(2).to_string())

fn, fp = d[d.kind == "FN"], d[d.kind == "FP"]
sc = f"{name}_score"
sets = [("FN random", fn.sample(6, random_state=config.SEED)), ("FN most confident", fn.nsmallest(6, sc)),
        ("FP random", fp.sample(6, random_state=config.SEED)), ("FP most confident", fp.nlargest(6, sc))]
fig, axes = plt.subplots(4, 6, figsize=(14, 9.5))
for r, (label, sub) in enumerate(sets):
    for c, row in enumerate(sub.itertuples()):
        ax = axes[r, c]
        ax.imshow(cv2.cvtColor(cv2.imread(str(config.ROOT / row.path)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row.slide_code} score {getattr(row, sc):.1f}", fontsize=7); ax.axis("off")
        if c == 0:
            ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(config.RESULTS / f"error_gallery_{name}.png", dpi=120)
