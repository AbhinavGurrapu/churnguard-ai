import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score
)
from src.models.preprocess import prepare_train_test_data

MODEL_SAVE_DIR = "src/models/saved_models"

def run_logistic_regression_baseline(csv_path="data/churn_feature_store.csv"):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 1. Load exact same 80/20 train/test split from preprocessing module
    data_dict = prepare_train_test_data(filepath=csv_path, test_size=0.20, random_state=42)
    
    X_train_lr = data_dict['X_train_lr']
    X_test_lr = data_dict['X_test_lr']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    
    # 2. Fit Logistic Regression Classifier
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_lr, y_train)
    
    # 3. Evaluate ONLY on Held-Out Test Set (600 users)
    y_pred_lr = lr_model.predict(X_test_lr)
    y_prob_lr = lr_model.predict_proba(X_test_lr)[:, 1]
    
    # Calculate Metrics
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn_l, fp_l, fn_l, tp_l = cm_lr.ravel()
    
    acc_l = accuracy_score(y_test, y_pred_lr)
    prec_l = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_l = recall_score(y_test, y_pred_lr, zero_division=0)
    f1_l = f1_score(y_test, y_pred_lr, zero_division=0)
    roc_auc_l = roc_auc_score(y_test, y_prob_lr)
    
    # Save Model Artifact
    model_path = os.path.join(MODEL_SAVE_DIR, "baseline_lr_pipeline.joblib")
    joblib.dump(lr_model, model_path)
    
    # Report Results
    print("=" * 70)
    print("   CHURNGUARD AI - LOGISTIC REGRESSION BASELINE MODEL EVALUATION  ")
    print("=" * 70)
    print(f"\n[1] CONFUSION MATRIX BREAKDOWN (Test Set = 600 users):")
    print(f"    +-------------------+-------------------+")
    print(f"    | TN: {tn_l:<13} | FP: {fp_l:<13} | (Actual Active: {tn_l+fp_l})")
    print(f"    +-------------------+-------------------+")
    print(f"    | FN: {fn_l:<13} | TP: {tp_l:<13} | (Actual Churned: {fn_l+tp_l})")
    print(f"    +-------------------+-------------------+")
    
    print(f"\n[2] EVALUATION METRICS:")
    print(f" - Precision:  {prec_l:.4f} ({prec_l*100:.2f}%)")
    print(f" - Recall:     {rec_l:.4f} ({rec_l*100:.2f}%)")
    print(f" - F1 Score:   {f1_l:.4f}")
    print(f" - ROC-AUC:    {roc_auc_l:.4f}")
    print(f" - Accuracy:   {acc_l:.4f} ({acc_l*100:.2f}%) [Reference Only]")
    
    print(f"\n[3] MODEL SAVED:")
    print(f" - Saved to: '{model_path}'")
    print("=" * 70)

if __name__ == "__main__":
    run_logistic_regression_baseline()
