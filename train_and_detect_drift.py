import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ALPHA = 0.05
EXPERIMENT = "Iris_MLflow_Drift_Analysis"

def main():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    # Reference = training data. Current data = test data with simulated shifts.
    reference = X_train.reset_index(drop=True)
    current = X_test.reset_index(drop=True).copy()
    current["sepal length (cm)"] += 0.8
    current["petal length (cm)"] += 1.0

    results = {}
    for feature in X.columns:
        stat, p = ks_2samp(reference[feature], current[feature])
        results[feature] = {"ks_statistic": float(stat), "p_value": float(p),
                            "drift_detected": bool(p < ALPHA)}
    drifted = [f for f, r in results.items() if r["drift_detected"]]
    report = {
        "dataset": "scikit-learn Iris", "records": len(X),
        "reference_data": "80% training split",
        "current_data": "20% test split with simulated feature shifts",
        "drift_test": "Two-sample Kolmogorov-Smirnov test",
        "alpha": ALPHA, "accuracy": float(accuracy),
        "weighted_f1_score": float(weighted_f1),
        "drifted_features": drifted, "feature_results": results,
        "overall_drift_detected": bool(drifted),
        "note": "Drift is simulated for this educational demonstration."
    }
    Path("drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="iris_random_forest_drift_check"):
        mlflow.log_params({"model_type": "RandomForestClassifier",
                           "n_estimators": 100, "max_depth": 5,
                           "test_size": 0.2, "random_state": 42,
                           "drift_test": "KS two-sample test", "alpha": ALPHA})
        mlflow.log_metrics({"accuracy": float(accuracy),
                            "weighted_f1_score": float(weighted_f1),
                            "number_of_drifted_features": len(drifted),
                            "overall_drift_detected": int(bool(drifted))})
        for feature, result in results.items():
            safe = feature.replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metrics({f"{safe}_ks_statistic": result["ks_statistic"],
                                f"{safe}_p_value": result["p_value"],
                                f"{safe}_drift_detected": int(result["drift_detected"])})
        mlflow.log_artifact("drift_report.json")
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    print(f"Iris records: {len(X)}")
    print(f"Accuracy: {accuracy:.4f}; weighted F1: {weighted_f1:.4f}")
    print(f"Drift detected: {bool(drifted)}; flagged features: {drifted}")
    print(f"MLflow experiment: {EXPERIMENT}")

if __name__ == "__main__":
    main()
