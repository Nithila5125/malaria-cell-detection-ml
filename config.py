from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
CLASS_MAP = {"Uninfected": 0, "Parasitized": 1}  # positive class = Parasitized
