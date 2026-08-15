import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.models.preprocess import prepare_train_test_data

MODEL_PATH = "src/models/saved_models/xgboost_model.joblib"
OUTPUT_DIR = "src/explainability"
LOCAL_CSV_PATH = "data/shap_local_explanations.csv"
LOCAL_JSON_PATH = "data/shap_local_explanations.json"

def generate_driver_explanation(feature, val, shap_val):
    """Deterministic Python logic to generate natural language driver explanations."""
    direction_str = "UP (+)" if shap_val > 0 else "DOWN (-)"
    formatted_shap = f"{shap_val:+.2f}"
    
    if feature == 'sessions_recent_14d':
        if shap_val > 0:
            return f"Recent 14-day activity dropped to {val:.0f} sessions (pushed churn risk UP by {formatted_shap})"
        else:
            return f"High recent activity with {val:.0f} sessions in last 14d (pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'support_ticket_count':
        if shap_val > 0:
            return f"Filed {val:.0f} support tickets in 60d (pushed churn risk UP by {formatted_shap})"
        else:
            return f"Low support friction with {val:.0f} tickets (pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'days_since_last_session':
        if shap_val > 0:
            return f"Inactive for {val:.0f} consecutive days (pushed churn risk UP by {formatted_shap})"
        else:
            return f"Recent session logged {val:.0f} days ago (pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'session_drop_pct':
        if shap_val > 0:
            return f"Session volume dropped by {val:.1f}% in final 14d (pushed churn risk UP by {formatted_shap})"
        else:
            return f"Stable/increasing activity trend ({val:.1f}% drop, pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'core_feature_usage_count':
        if shap_val > 0:
            return f"Low adoption of core features ({val:.0f} events, pushed churn risk UP by {formatted_shap})"
        else:
            return f"Strong adoption of core features ({val:.0f} events, pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'unresolved_tickets':
        if shap_val > 0:
            return f"Has {val:.0f} unresolved support tickets (pushed churn risk UP by {formatted_shap})"
        else:
            return f"No open unresolved tickets (pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'total_events_60d':
        if shap_val > 0:
            return f"Low overall engagement depth ({val:.0f} total events, pushed churn risk UP by {formatted_shap})"
        else:
            return f"High overall engagement depth ({val:.0f} total events, pulled churn risk DOWN by {formatted_shap})"
            
    elif feature == 'avg_session_gap_days':
        if pd.isnull(val):
            return f"Insufficient sessions to compute gap (pushed churn risk UP by {formatted_shap})"
        if shap_val > 0:
            return f"High average session gap of {val:.1f} days (pushed churn risk UP by {formatted_shap})"
        else:
            return f"Frequent logins with average gap of {val:.1f} days (pulled churn risk DOWN by {formatted_shap})"
            
    else:
        return f"{feature} = {val} (pushed churn risk {direction_str} by {formatted_shap})"

def run_local_shap_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load trained XGBoost model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained XGBoost model not found at '{MODEL_PATH}'. Please run Day 2 training first.")
    
    xgb_model = joblib.load(MODEL_PATH)
    
    # 2. Load test set and raw feature store to retrieve user_ids
    df_raw = pd.read_csv("data/churn_feature_store.csv")
    data_dict = prepare_train_test_data(filepath="data/churn_feature_store.csv", test_size=0.20, random_state=42)
    
    X_test_xgb = data_dict['X_test_xgb']
    y_test = data_dict['y_test']
    
    test_user_ids = df_raw.loc[X_test_xgb.index, 'user_id'].values
    
    print("=" * 75)
    print("        CHURNGUARD AI - DAY 3 STEP 2 LOCAL / PER-USER SHAP EXPLANATIONS        ")
    print("=" * 75)

    # 3. Compute SHAP values for test set users
    explainer = shap.TreeExplainer(xgb_model)
    shap_explanation = explainer(X_test_xgb)
    
    vals = shap_explanation.values if hasattr(shap_explanation, "values") else shap_explanation
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]

    # 4. Predict probabilities & hard predictions
    y_prob = xgb_model.predict_proba(X_test_xgb)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    # 5. Extract Per-User Top 3 SHAP Drivers
    local_records = []

    for i in range(len(X_test_xgb)):
        uid = int(test_user_ids[i])
        actual_churn = int(y_test.iloc[i])
        pred_prob = float(y_prob[i])
        pred_class = int(y_pred[i])
        
        user_features = X_test_xgb.iloc[i]
        user_shap = vals[i]
        
        # Rank features by absolute SHAP value for this user
        abs_shap = np.abs(user_shap)
        top_3_indices = np.argsort(abs_shap)[::-1][:3]
        
        top_drivers = []
        for idx in top_3_indices:
            feat_name = X_test_xgb.columns[idx]
            feat_val = user_features[feat_name]
            s_val = float(user_shap[idx])
            
            explanation_text = generate_driver_explanation(feat_name, feat_val, s_val)
            
            top_drivers.append({
                'feature': feat_name,
                'feature_value': float(feat_val) if not pd.isnull(feat_val) else None,
                'shap_value': s_val,
                'direction': 'UP (+)' if s_val > 0 else 'DOWN (-)',
                'explanation': explanation_text
            })
            
        record = {
            'user_id': uid,
            'actual_churn_label': actual_churn,
            'predicted_churn_probability': round(pred_prob, 4),
            'predicted_class_0_50': pred_class,
            'base_value_log_odds': round(float(base_val), 4),
            'top_driver_1': top_drivers[0],
            'top_driver_2': top_drivers[1],
            'top_driver_3': top_drivers[2]
        }
        local_records.append(record)

    # 6. Save structured per-user explanations to CSV and JSON
    df_local = pd.DataFrame([
        {
            'user_id': r['user_id'],
            'actual_churn': r['actual_churn_label'],
            'churn_probability': r['predicted_churn_probability'],
            'predicted_class': r['predicted_class_0_50'],
            'driver_1_feature': r['top_driver_1']['feature'],
            'driver_1_shap': r['top_driver_1']['shap_value'],
            'driver_1_text': r['top_driver_1']['explanation'],
            'driver_2_feature': r['top_driver_2']['feature'],
            'driver_2_shap': r['top_driver_2']['shap_value'],
            'driver_2_text': r['top_driver_2']['explanation'],
            'driver_3_feature': r['top_driver_3']['feature'],
            'driver_3_shap': r['top_driver_3']['shap_value'],
            'driver_3_text': r['top_driver_3']['explanation']
        }
        for r in local_records
    ])
    
    df_local.to_csv(LOCAL_CSV_PATH, index=False)
    with open(LOCAL_JSON_PATH, 'w') as f:
        json.dump(local_records, f, indent=2)

    print(f"\n[1] LOCAL PER-USER SHAP EXPLANATIONS SAVED TO:")
    print(f" - CSV:  '{LOCAL_CSV_PATH}'")
    print(f" - JSON: '{LOCAL_JSON_PATH}'")

    # 7. Select High-Risk Example User for Inspection (Highest Churn Probability)
    high_risk_idx = np.argmax(y_prob)
    high_risk_record = local_records[high_risk_idx]
    
    print(f"\n[2] EXAMPLE HIGH-RISK USER PROFILE (USER #{high_risk_record['user_id']}):")
    print(f" - User ID:                      {high_risk_record['user_id']}")
    print(f" - Predicted Churn Probability:  {high_risk_record['predicted_churn_probability']*100:.2f}% ({high_risk_record['predicted_churn_probability']})")
    print(f" - Predicted Class (0.50 Threshold): {high_risk_record['predicted_class_0_50']} (HIGH RISK)")
    print(f" - Actual Churn Label:           {high_risk_record['actual_churn_label']} (True Churner)")
    print(f"\n --- Top 3 SHAP Risk Drivers for User #{high_risk_record['user_id']} ---")
    for d_idx, d in enumerate([high_risk_record['top_driver_1'], high_risk_record['top_driver_2'], high_risk_record['top_driver_3']], 1):
        print(f"  Driver {d_idx}: [{d['feature']}] = {d['feature_value']}")
        print(f"    -> SHAP Contribution: {d['shap_value']:+.4f} ({d['direction']})")
        print(f"    -> Explanation:       \"{d['explanation']}\"\n")

    # 8. Generate Local SHAP Waterfall Plot for the High-Risk Example User
    plt.figure(figsize=(10, 6))
    user_exp = shap_explanation[high_risk_idx]
    shap.plots.waterfall(user_exp, max_display=7, show=False)
    
    example_plot_path = f"src/explainability/shap_user_{high_risk_record['user_id']}_waterfall.png"
    plt.title(f"ChurnGuard AI - SHAP Local Risk Explanation for User #{high_risk_record['user_id']}", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(example_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[3] LOCAL SHAP WATERFALL PLOT SAVED TO:")
    print(f" - '{example_plot_path}'")
    print("=" * 75)

if __name__ == "__main__":
    run_local_shap_analysis()
