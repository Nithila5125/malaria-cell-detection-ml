import hashlib, sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import config

out = config.PROCESSED / "split.csv"
if out.exists():
    sys.exit(f"{out} already exists. The split is frozen; delete it only if you intend to restart modelling.")

df = pd.read_csv(config.RESULTS / "dataset_inventory.csv")
assert not df.corrupt.any() and df.slide_code.notna().all()
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
_, test_idx = next(sgkf.split(df, df.y, groups=df.slide_code))
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

tr, te = df[df.split == "train"], df[df.split == "test"]
assert set(tr.slide_code).isdisjoint(te.slide_code)
config.PROCESSED.mkdir(parents=True, exist_ok=True)
df[["path", "name", "label", "y", "slide_code", "split"]].to_csv(out, index=False)
(config.PROCESSED / "split.sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest())

print(df.groupby("split").label.value_counts().unstack())
print(df.groupby("split").slide_code.nunique().rename("slide codes"))
print("Test share of images: %.3f" % (len(te) / len(df)))
print("Test codes only Uninfected:", int((te.groupby("slide_code").y.max() == 0).sum()))
