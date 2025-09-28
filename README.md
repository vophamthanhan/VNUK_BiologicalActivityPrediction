# Biological Activity Prediction

An interactive Streamlit application and supporting machine learning assets for predicting the biological activity of compounds sourced from ChEMBL data. The project bundles cleaned datasets, pre-trained models (Gaussian Naive Bayes, XGBoost, and an Artificial Neural Network), and notebooks that document the full training and evaluation workflow.

## Features
- Pre-trained models ready for immediate inference through the Streamlit UI.
- Batch predictions from user-supplied CSV files with automatic preprocessing and normalization.
- Visual insights including distribution plots and descriptive statistics for prediction confidences.
- Reproducible notebooks covering data preprocessing, model training, and comparative evaluation.
- Sample CSV files and a demo video to help you get started quickly.

## Repository Layout
```text
.
|- data/                  # Raw and split datasets derived from ChEMBL
|- models/                # Saved models, scaler, and development notebooks
|- test/                  # Example CSV inputs for quick experimentation
|- webapp/app.py          # Streamlit application entry point
`- DemoVideo.mp4          # Walkthrough of the deployed application
```

## Requirements
- Python 3.9 or newer
- Recommended packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `tensorflow`, `matplotlib`, `seaborn`, `xgboost`

Create and activate a virtual environment (optional but encouraged) and install dependencies:

```bash
python -m venv .venv
# Windows
. .venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install streamlit pandas numpy scikit-learn joblib tensorflow matplotlib seaborn xgboost
```

If you maintain a `requirements.txt`, you can install everything at once with `pip install -r requirements.txt`.

## Running the Streamlit App
1. Ensure the model files in `models/` are accessible to the app. The current script loads them via absolute paths; update `webapp/app.py` so that `MODEL_DIR` points to your local `models` folder, for example:
   ```python
   from pathlib import Path

   MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
   scaler = joblib.load(MODEL_DIR / "scaler.pkl")
   gnb_model = joblib.load(MODEL_DIR / "gaussian_nb.pkl")
   xgb_model = joblib.load(MODEL_DIR / "xgboost.pkl")
   ann_model = load_model(MODEL_DIR / "ann.h5", compile=False)
   ```
2. Launch the app:
   ```bash
   streamlit run webapp/app.py
   ```
3. Open the URL shown in the terminal (defaults to `http://localhost:8501`).

## Input Data Format
Your CSV file should at minimum include the columns shown below (see `test/sample*.csv` for ready-made examples):

| Column      | Description                                               |
|-------------|-----------------------------------------------------------|
| `Compounds` | Numeric representation of compound-related measurements   |
| `Type`      | Categorical type (e.g., CELL-LINE, SMALL MOLECULE, etc.)  |
| `Tax ID`    | NCBI taxonomy identifier for the organism                 |
| `Organism`  | Organism name matching the taxonomy identifier            |

Missing values are automatically filled with column means, categorical fields are label-encoded on the fly, and numeric columns are scaled using the stored `scaler.pkl`.

## Notebooks
The `models/` directory contains three Jupyter notebooks that document the end-to-end pipeline:
- `data_preprocessing.ipynb` - Cleaning, feature engineering, and train/test splitting.
- `model_training.ipynb` - Training GaussianNB, XGBoost, and ANN models.
- `evaluate_models.ipynb` - Benchmarking performance metrics and visual diagnostics.

Run them in order within a Jupyter environment if you need to retrain or extend the models.

## Demo & Testing Assets
- `DemoVideo.mp4` showcases the user experience and typical workflow.
- `test/` includes several CSV files you can upload directly into the app to verify predictions.

## Troubleshooting
- **Model path errors:** Ensure the absolute or relative paths in `webapp/app.py` match your local directory layout.
- **Missing dependencies:** Re-run the `pip install` command above and confirm your virtual environment is active.
- **Streamlit port conflicts:** Use `streamlit run webapp/app.py --server.port 8502` to switch to a different port.

## Next Steps
Possible extensions include integrating a REST API, refining feature engineering, or packaging the project with Docker for easier deployment.
