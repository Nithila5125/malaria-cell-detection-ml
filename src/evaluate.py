import sys, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
import config

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
d = pd.read_parquet(config.PROCESSED / "features.parquet")
te = d[d.split == "test"].reset_index(drop=True)
y, codes = te.y.to_numpy(), te.slide_code.to_numpy()
by_code = {c: np.where(codes == c)[0] for c in np.unique(codes)}
keys = list(by_code)
rng = np.random.default_rng(config.SEED)
boot = [np.concatenate([by_code[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(500)]
print(mode, "| test rows", len(te), "| test slide codes", len(keys))

def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]

labels = ["Uninfected", "Parasitized"]
rows, preds = [], te[["path", "name", "label", "y", "slide_code"]].copy()
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
fig2, ax2 = plt.subplots(figsize=(5, 4.5))
for k, name in enumerate(["lr", "rf", "svm"]):
    art = joblib.load(config.MODELS / f"{mode}_{name}.joblib")
    m, cols = art["model"], art["cols"]
    X = te[cols]; s = score(m, X); p = m.predict(X)
    cm = confusion_matrix(y, p, labels=[0, 1])
    aucs = [roc_auc_score(y[i], s[i]) for i in boot]
    auc = roc_auc_score(y, s)
    rows.append(dict(model=name, features=mode, accuracy=accuracy_score(y, p),
                     precision=precision_score(y, p), recall=recall_score(y, p), f1=f1_score(y, p),
                     roc_auc=auc, auc_ci_lo=np.percentile(aucs, 2.5), auc_ci_hi=np.percentile(aucs, 97.5),
                     tn=cm[0, 0], fp=cm[0, 1], fn=cm[1, 0], tp=cm[1, 1]))
    preds[f"{name}_score"], preds[f"{name}_pred"] = s, p
    ax[k].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[k].text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[k].set_xticks([0, 1], labels, fontsize=8); ax[k].set_yticks([0, 1], labels, fontsize=8)
    ax[k].set_xlabel("Predicted"); ax[k].set_ylabel("True"); ax[k].set_title(name.upper())
    fpr, tpr, _ = roc_curve(y, s)
    ax2.plot(fpr, tpr, label=f"{name.upper()} (AUC {auc:.3f})")
ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate (Parasitized)")
ax2.set_title(f"ROC, held-out test set ({mode} features)"); ax2.legend(loc="lower right")
fig.tight_layout(); fig.savefig(config.RESULTS / f"cm_{mode}.png", dpi=130)
fig2.tight_layout(); fig2.savefig(config.RESULTS / f"roc_{mode}.png", dpi=130)
r = pd.DataFrame(rows)
r.to_csv(config.RESULTS / f"test_metrics_{mode}.csv", index=False)
preds.to_csv(config.RESULTS / f"test_predictions_{mode}.csv", index=False)
print(r.round(4).to_string(index=False))
