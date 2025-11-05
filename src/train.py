import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve

# ---- MLflow wiring ----
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")  # <- no localhost fallback, must be set from env
if not MLFLOW_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI must be set (e.g., http://127.0.0.1:5050 for port-forward).")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)

# When server is started with --serve-artifacts + --artifacts-destination, no client creds needed
os.environ["MLFLOW_AZURE_STORAGE_AUTH_TYPE"] = "MSI"  # harmless if not used client-side

EXPERIMENT_NAME = "Stroke_Prediction_LightGBM_TelemetryGuard_v2"  
MODEL_NAME = "TelemetryGuard_Stroke_Model"
PROC_PATH = "data/processed"
BASELINE_PATH = "data/baselines"

mlflow.set_experiment(EXPERIMENT_NAME)

# Load processed data
X_train = pd.read_csv(f"{PROC_PATH}/X_train.csv")
X_test  = pd.read_csv(f"{PROC_PATH}/X_test.csv")
y_train = pd.read_csv(f"{PROC_PATH}/y_train.csv").values.ravel()
y_test  = pd.read_csv(f"{PROC_PATH}/y_test.csv").values.ravel()


def plot_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path):
    importance = model.feature_importance()
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importance[indices])
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_log():
    # Handle imbalance
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": pos_weight,
        "seed": 42
    }

    with mlflow.start_run(run_name="LightGBM_Stroke_Prediction"):
        # Split validation
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200)

        # Predict
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

        # --- Choose optimal threshold on the VALIDATION set (not test) ---
        from sklearn.metrics import precision_recall_curve
        
        # Probabilities on val & test
        y_proba_val  = model.predict(X_val,  num_iteration=getattr(model, "best_iteration", None))
        y_proba_test = y_pred_proba  # already computed for X_test
        
        # precision_recall_curve returns: precision (len=N+1), recall (len=N+1), thresholds (len=N)
        prec, rec, thr = precision_recall_curve(y_val, y_proba_val)
        
        # Compute F1 only where a threshold exists (align lengths with `thr`)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_vals = 2.0 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1])
            f1_vals = np.nan_to_num(f1_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fallback if no thresholds (degenerate case)
        best_thr = float(thr[np.argmax(f1_vals)]) if thr.size > 0 else 0.5
        
        # --- Convert test probabilities to binary labels using the chosen threshold ---
        y_pred = (y_proba_test >= best_thr).astype(int)
        
        # --- Metrics (labels for cls metrics, probs for AUC) ---
        accuracy  = accuracy_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba_test)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        
        # --- Log to MLflow ---
        mlflow.log_params(params)
        mlflow.log_metric("accuracy",  accuracy)
        mlflow.log_metric("roc_auc",   roc_auc)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("best_threshold", best_thr)
        
        # --- Artifacts (use LABELS for confusion matrix) ---
        os.makedirs("artifacts", exist_ok=True)
        plot_confusion(y_test, y_pred, "artifacts/confusion_matrix.png")
        plot_feature_importance(model, X_train.columns, "artifacts/feature_importance.png")
        #mlflow.log_artifact("artifacts/confusion_matrix.png",   artifact_path="plots")
        #mlflow.log_artifact("artifacts/feature_importance.png", artifact_path="plots")

        # Register model
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        print(f"Model logged and registered under name: {MODEL_NAME}")
        print(f"Run successfully completed. AUC={roc_auc:.3f}, F1={f1:.3f}, thr={best_thr:.3f}")


if __name__ == "__main__":
    train_and_log()