import re, sys, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import config

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
name = sys.argv[2] if len(sys.argv) > 2 else "lr"
N = 30
d = pd.read_parquet(config.PROCESSED / "features.parquet")
te = d[d.split == "test"].reset_index(drop=True)
art = joblib.load(config.MODELS / f"{mode}_{name}.joblib")
m, cols = art["model"], art["cols"]
X, y = te[cols], te.y.to_numpy()

def grp(c):
    return "color" if re.match(r"^[RGBHSV]_", c) else "texture" if c.startswith("glcm_") else "morphology"

def score(est, Z):
    return est.decision_function(Z) if hasattr(est, "decision_function") else est.predict_proba(Z)[:, 1]

base = roc_auc_score(y, score(m, X))
res = permutation_importance(m, X, y, scoring="roc_auc", n_repeats=N, random_state=config.SEED, n_jobs=2)
f = pd.DataFrame(dict(feature=cols, group=[grp(c) for c in cols],
                      auc_drop=res.importances_mean, auc_drop_std=res.importances_std)).sort_values("auc_drop", ascending=False)
f.to_csv(config.RESULTS / f"perm_features_{mode}_{name}.csv", index=False)

rng = np.random.default_rng(config.SEED)
g_rows = []
for g in sorted(set(f.group)):
    gc = [c for c in cols if grp(c) == g]
    drops = []
    for _ in range(N):
        Xp = X.copy()
        Xp[gc] = X[gc].to_numpy()[rng.permutation(len(X))]
        drops.append(base - roc_auc_score(y, score(m, Xp)))
    g_rows.append(dict(group=g, n_features=len(gc), auc_drop=np.mean(drops), auc_drop_std=np.std(drops)))
gdf = pd.DataFrame(g_rows).sort_values("auc_drop", ascending=False)
gdf.to_csv(config.RESULTS / f"perm_groups_{mode}_{name}.csv", index=False)

top = f.head(15).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[2, 1]))
ax[0].barh(top.feature, top.auc_drop, xerr=top.auc_drop_std, color="#3b6ea5")
ax[0].set_xlabel("Drop in test ROC-AUC when permuted"); ax[0].set_title(f"Top 15 features ({name.upper()}, {mode})")
gg = gdf.iloc[::-1]
ax[1].barh(gg.group, gg.auc_drop, xerr=gg.auc_drop_std, color="#c4793a")
ax[1].set_xlabel("Drop in test ROC-AUC (whole group permuted)"); ax[1].set_title("Feature groups")
fig.tight_layout(); fig.savefig(config.RESULTS / f"perm_{mode}_{name}.png", dpi=130)
print("baseline test AUC %.4f | repeats %d" % (base, N))
print(f.head(12).round(4).to_string(index=False))
print(gdf.round(4).to_string(index=False))
