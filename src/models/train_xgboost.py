import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score
)
from src.models.preprocess import prepare_train_test_data

MODEL_SAVE_DIR = "src/models/saved_models"

def run_xgboost_training(csv_path="data/churn_feature_store.csv"):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. Load exact same 80/20 train/test split from preprocessing module
    data_dict = prepare_train_test_data(filepath=csv_path, test_size=0.20, random_state=42)
    
    X_train_xgb = data_dict['X_train_xgb']
    X_test_xgb = data_dict['X_test_xgb']
    X_test_lr = data_dict['X_test_lr']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    # 2. Instantiate XGBoost Classifier with explicit default configuration
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 3. Fit XGBoost Model strictly on Training Set (retaining native np.nan for gap days)
    xgb_model.fit(X_train_xgb, y_train)
    
    # 4. Predict on Held-Out Test Set (600 users)
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    y_prob_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
    
    # Calculate XGBoost Metrics
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    tn_x, fp_x, fn_x, tp_x = cm_xgb.ravel()
    
    acc_x = accuracy_score(y_test, y_pred_xgb)
    prec_x = precision_score(y_test, y_pred_xgb, zero_division=0)
    rec_x = recall_score(y_test, y_pred_xgb, zero_division=0)
    f1_x = f1_score(y_test, y_pred_xgb, zero_division=0)
    roc_auc_x = roc_auc_score(y_test, y_prob_xgb)
    
    # Save XGBoost Model Artifact
    xgb_save_path = os.path.join(MODEL_SAVE_DIR, "xgboost_model.joblib")
    joblib.dump(xgb_model, xgb_save_path)
    
    # 5. Load Baseline Logistic Regression for Side-by-Side Comparison
    lr_save_path = os.path.join(MODEL_SAVE_DIR, "baseline_lr_pipeline.joblib")
    if os.path.exists(lr_save_path):
        lr_pipeline = joblib.load(lr_save_path)
        y_pred_lr = lr_pipeline.predict(X_test_lr)
        y_prob_lr = lr_pipeline.predict_proba(X_test_lr)[:, 1]
        
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        tn_l, fp_l, fn_l, tp_l = cm_lr.ravel()
        
        acc_l = accuracy_score(y_test, y_pred_lr)
        prec_l = precision_score(y_test, y_pred_lr, zero_division=0)
        rec_l = recall_score(y_test, y_pred_lr, zero_division=0)
        f1_l = f1_score(y_test, y_pred_lr, zero_division=0)
        roc_auc_l = roc_auc_score(y_test, y_prob_lr)
    else:
        tn_l = fp_l = fn_l = tp_l = prec_l = rec_l = f1_l = roc_auc_l = acc_l = 0.0

    # Report Output
    print("=" * 75)
    print("      CHURNGUARD AI - XGBOOST VS LOGISTIC REGRESSION EVALUATION      ")
    print("=" * 75)
    
    print(f"\n[1] CONFUSION MATRIX COMPARISON (Held-Out Test Set = 600 Users):")
    print(f"    Logistic Regression Baseline:              XGBoost Classifier:")
    print(f"    +-------------------+-------------------+  +-------------------+-------------------+")
    print(f"    | TN: {tn_l:<13} | FP: {fp_l:<13} |  | TN: {tn_x:<13} | FP: {fp_x:<13} |")
    print(f"    +-------------------+-------------------+  +-------------------+-------------------+")
    print(f"    | FN: {fn_l:<13} | TP: {tp_l:<13} |  | FN: {fn_x:<13} | TP: {tp_x:<13} |")
    print(f"    +-------------------+-------------------+  +-------------------+-------------------+")
    
    print(f"\n[2] MODEL PERFORMANCE COMPARISON TABLE:")
    print(f"    +-----------------------+---------------------+---------------------+---------------------+")
    print(f"    | Metric                | Logistic Regression | XGBoost Classifier  | Improvement         |")
    print(f"    +-----------------------+---------------------+---------------------+---------------------+")
    print(f"    | ROC-AUC               | {roc_auc_l:<19.4f} | {roc_auc_x:<19.4f} | {roc_auc_x - roc_auc_l:+19.4f} |")
    print(f"    | Precision             | {prec_l:<19.4f} | {prec_x:<19.4f} | {prec_x - prec_l:+19.4f} |")
    print(f"    | Recall                | {rec_l:<19.4f} | {rec_x:<19.4f} | {rec_x - rec_l:+19.4f} |")
    print(f"    | F1 Score              | {f1_l:<19.4f} | {f1_x:<19.4f} | {f1_x - f1_l:+19.4f} |")
    print(f"    | Accuracy (Reference)  | {acc_l:<19.4f} | {acc_x:<19.4f} | {acc_x - acc_l:+19.4f} |")
    print(f"    +-----------------------+---------------------+---------------------+---------------------+")

    print(f"\n[3] MODEL ARTIFACT SAVED:")
    print(f" - Saved to: '{xgb_save_path}'")
    print("=" * 75)

if __name__ == "__main__":
    run_xgboost_training()
