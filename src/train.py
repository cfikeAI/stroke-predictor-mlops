import pandas as pd
import os
import json
import mlflow
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#PATHS
PROC_PATH = "data/processed"
BASELINE_PATH = "data/baselines"

# Load processed data
X_train = pd.read_csv(f"{PROC_PATH}/X_train.csv")
X_test = pd.read_csv(f"{PROC_PATH}/X_test.csv")
y_train = pd.read_csv(f"{PROC_PATH}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{PROC_PATH}/y_test.csv").values.ravel()

#MLFlow
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Stroke_Prediction_LightGBM_TelemetryGuard")

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
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 42
    }

    with mlflow.start_run(run_name = "LightGBM_Stroke_Prediction"):
        #train model
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)

        model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

        # Predictions and evaluation
        y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters, metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        # Log artifacts
        os.makedirs("artifacts", exist_ok=True)
        conf_path = "artifacts/confusion_matrix.png"
        featimp_path = "artifacts/feature_importance.png"

        plot_confusion(y_test, y_pred, conf_path)
        plot_feature_importance(model, X_train.columns, featimp_path)

        mlflow.log_artifact(conf_path)
        mlflow.log_artifact(featimp_path)

        # Log baselines (for drift reference)
        if os.path.exists(BASELINE_PATH):
            mlflow.log_artifact(BASELINE_PATH)

        #log model
        mlflow.lightgbm.log_model(model, "model")

        print("Run successfully completed. AUC={auc:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    train_and_log()