import re, sys, time
import joblib, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import config

mode = sys.argv[1] if len(sys.argv) > 1 else "all"
assert mode in ("all", "nocolor")
d = pd.read_parquet(config.PROCESSED / "features.parquet")
tr = d[d.split == "train"]
cols = list(d.columns[6:])
if mode == "nocolor":
    cols = [c for c in cols if not re.match(r"^[RGBHSV]_", c)]
X, y, g = tr[cols], tr.y, tr.slide_code
print(mode, "| train rows", len(tr), "| features", len(cols))

specs = {
    "lr": (Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=3000, random_state=config.SEED))]),
           {"m__C": [0.01, 0.1, 1, 10, 100]}, 5),
    "rf": (RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=1),
           {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]}, 8),
    "svm": (Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", cache_size=1000, random_state=config.SEED))]),
            {"m__C": [0.5, 2, 8], "m__gamma": ["scale", 0.01, 0.03]}, 5),
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
config.MODELS.mkdir(exist_ok=True); config.RESULTS.mkdir(exist_ok=True)
rows = []
for name, (est, grid, n_iter) in specs.items():
    t = time.time()
    s = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv, n_jobs=2, refit="roc_auc",
                           scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
                           random_state=config.SEED, verbose=2)
    s.fit(X, y, groups=g)
    b = s.best_index_
    rows.append(dict(model=name, features=mode, cv_auc=s.cv_results_["mean_test_roc_auc"][b],
                     cv_auc_std=s.cv_results_["std_test_roc_auc"][b], cv_f1=s.cv_results_["mean_test_f1"][b],
                     cv_acc=s.cv_results_["mean_test_accuracy"][b], best_params=str(s.best_params_),
                     seconds=round(time.time() - t)))
    joblib.dump(dict(model=s.best_estimator_, cols=cols), config.MODELS / f"{mode}_{name}.joblib")
    print(rows[-1], flush=True)
r = pd.DataFrame(rows)
r.to_csv(config.RESULTS / f"cv_{mode}.csv", index=False)
(config.RESULTS / f"primary_model_{mode}.txt").write_text(r.sort_values("cv_auc").iloc[-1].model)
print(r[["model", "cv_auc", "cv_auc_std", "cv_f1", "cv_acc", "seconds"]].round(4).to_string(index=False))
print("Primary model (highest CV AUC):", r.sort_values("cv_auc").iloc[-1].model)
