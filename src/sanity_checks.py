import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import config

d = pd.read_parquet(config.PROCESSED / "features.parquet")
tr, te = d[d.split == "train"], d[d.split == "test"]
art = joblib.load(config.MODELS / "all_lr.joblib")
m, cols = clone(art["model"]), art["cols"]
rng = np.random.default_rng(config.SEED)
m.fit(tr[cols], rng.permutation(tr.y.to_numpy()))
print("Shuffled-label LR test AUC: %.3f (expect about 0.5)" % roc_auc_score(te.y, m.decision_function(te[cols])))
dm = DummyClassifier(strategy="most_frequent").fit(tr[cols], tr.y)
print("Majority-class test accuracy: %.3f" % accuracy_score(te.y, dm.predict(te[cols])))
