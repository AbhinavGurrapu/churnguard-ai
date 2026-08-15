import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.models.preprocess import prepare_train_test_data

MODEL_PATH = "src/models/saved_models/xgboost_model.joblib"
OUTPUT_DIR = "src/explainability"
DATA_OUTPUT_PATH = "data/shap_global_importance.csv"
PLOT_OUTPUT_PATH = "src/explainability/shap_global_bar_plot.png"

def run_global_shap_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load trained XGBoost model artifact
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained XGBoost model not found at '{MODEL_PATH}'. Please run Day 2 training first.")
    
    xgb_model = joblib.load(MODEL_PATH)
    
    # 2. Load held-out test data split (600 test users)
    data_dict = prepare_train_test_data(filepath="data/churn_feature_store.csv", test_size=0.20, random_state=42)
    X_test_xgb = data_dict['X_test_xgb']
    
    print("=" * 75)
    print("        CHURNGUARD AI - DAY 3 STEP 1 GLOBAL SHAP EXPLAINABILITY        ")
    print("=" * 75)
    
    # 3. Instantiate SHAP TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_xgb)
    
    # Handle SHAP Explanation object values matrix
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values

    # Base value extraction
    if hasattr(explainer, "expected_value"):
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = base_val[0]
    else:
        base_val = np.nan

    print(f"\n[1] SHAP EXPLAINER INITIALIZATION:")
    print(f" - Explainer Type:  TreeExplainer (Optimized for XGBoost)")
    print(f" - Base Value:       {base_val:.4f} (Raw log-odds baseline prediction)")
    print(f" - Test Sample Size: {X_test_xgb.shape[0]} users across {X_test_xgb.shape[1]} features")

    # 4. Compute Global Feature Importance (Mean Absolute SHAP Value)
    mean_abs_shap = np.abs(vals).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': X_test_xgb.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    
    importance_df['rank'] = importance_df.index + 1
    
    # Save structured global importance table to data CSV
    importance_df.to_csv(DATA_OUTPUT_PATH, index=False)
    print(f"\n[2] GLOBAL FEATURE IMPORTANCE SAVED TO:")
    print(f" - '{DATA_OUTPUT_PATH}'")

    # 5. Display Top Global Feature Drivers
    print(f"\n[3] TOP GLOBAL CHURN DRIVERS (RANKED BY MEAN |SHAP| IMPACT):")
    print(f"    +------+---------------------------------+-----------------+")
    print(f"    | Rank | Feature Name                    | Mean |SHAP|     |")
    print(f"    +------+---------------------------------+-----------------+")
    for _, row in importance_df.iterrows():
        print(f"    | {int(row['rank']):<4} | {row['feature']:<31} | {row['mean_abs_shap']:<15.4f} |")
    print(f"    +------+---------------------------------+-----------------+")

    # 6. Generate and Save Global SHAP Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_test_xgb, plot_type="bar", show=False)
    plt.title("ChurnGuard AI - Global SHAP Feature Importance", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[4] SHAP GLOBAL BAR PLOT SAVED TO:")
    print(f" - '{PLOT_OUTPUT_PATH}'")
    print("=" * 75)

if __name__ == "__main__":
    run_global_shap_analysis()
