# Explainable malaria-infected cell detection (classical ML)

Classifies Parasitized vs Uninfected red blood cells from the NIH malaria cell images (Kaggle version) using
morphology, color and GLCM texture features with Logistic Regression, Random Forest and SVM.

## Setup
Python 3.13.2 on Windows. `python -m venv .venv`, activate it, then `pip install -r requirements.txt`
(exact versions in `requirements.lock.txt`). Download the dataset from
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria into `data/raw/` so that exactly one
`Parasitized` and one `Uninfected` folder exist under `data/raw/` (remove the nested duplicate `cell_images/cell_images`).

## Run order (from the project root)
    python -m src.inspect_data
    python -m src.sample_images
    python -m src.mask_check
    python -m src.make_split
    python -m src.extract_features
    python -m src.train_models all
    python -m src.evaluate all
    python -m src.train_models nocolor
    python -m src.evaluate nocolor
    python -m src.perm_importance all lr
    python -m src.error_analysis

Seed is 42 (`config.py`). Positive class is Parasitized. All numbers are written to `results/`.

## Design notes and limitations
- Cells are supplied already cut out on a black background; the mask is the non-black region (`src/masking.py`). Morphology therefore describes the dataset's own masks; area and perimeter are in pixels, measured before any resizing.
- No official patient ID is documented. The filename code (e.g. C100, 200 codes) is used as a patient proxy: the frozen train/test split is grouped by this code (`data/processed/split.csv`), but it is not a verified patient-level split.
- Model tuning used training data only (5-fold grouped CV, ROC-AUC). The test set was evaluated once; the color ablation, permutation importance and error analysis are diagnostic and were not used to retune.
- The GLCM texture is computed on grayscale intensity, so the no-color ablation still contains intensity information.
- Single dataset and site; not validated for clinical use.
