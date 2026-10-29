# Iris MLflow Drift Analysis

Uses the built-in scikit-learn Iris dataset; no dataset download is needed.
Trains a Random Forest classifier, logs accuracy/F1 and the model to MLflow,
and tests feature distribution drift with the two-sample KS test. Drift is
simulated by shifting sepal length and petal length in the current/test data.

## Windows commands
```bat
python --version
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_and_detect_drift.py
python -m mlflow ui --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000. Keep the MLflow UI terminal open.
A p-value below 0.05 flags a feature. Drift does not by itself prove that
model performance has degraded. The shifted data is simulated, not production data.
