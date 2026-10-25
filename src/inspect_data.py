import hashlib, re, sys
import cv2, pandas as pd
from tqdm import tqdm
import config

dirs = {k: [p for p in config.RAW.rglob(k) if p.is_dir()] for k in config.CLASS_MAP}
for k, v in dirs.items():
    if len(v) != 1:
        sys.exit(f"Expected 1 '{k}' folder under data/raw, found {len(v)}: {v}. Remove the duplicate.")

rows = []
for label, [d] in dirs.items():
    for p in tqdm(sorted(d.glob("*.png")), desc=label):
        img = cv2.imread(str(p))
        m = re.match(r"^(C\d+)", p.name)
        rows.append(dict(
            path=str(p.relative_to(config.ROOT)), name=p.name, label=label,
            y=config.CLASS_MAP[label], corrupt=img is None,
            h=None if img is None else img.shape[0],
            w=None if img is None else img.shape[1],
            md5=hashlib.md5(p.read_bytes()).hexdigest(),
            slide_code=m.group(1) if m else None))

df = pd.DataFrame(rows)
config.RESULTS.mkdir(exist_ok=True)
df.to_csv(config.RESULTS / "dataset_inventory.csv", index=False)

print("\nCounts:\n", df.label.value_counts())
print("Corrupt:", int(df.corrupt.sum()))
print("Size min/max (h,w):", df.h.min(), df.h.max(), df.w.min(), df.w.max())
print("Exact duplicate files:", int(df.md5.duplicated(keep=False).sum()))
print("Duplicates across classes:", int(df.groupby("md5").label.nunique().gt(1).sum()))
print("Filenames with slide code:", int(df.slide_code.notna().sum()), "of", len(df))
g = df.dropna(subset=["slide_code"]).groupby("slide_code").label.nunique()
print("Unique slide codes:", df.slide_code.nunique(), "| codes in both classes:", int(g.gt(1).sum()))
print(df.name.head(3).tolist(), df[df.label == "Uninfected"].name.head(3).tolist())
