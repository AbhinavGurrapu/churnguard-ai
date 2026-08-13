import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RAW_FEATURE_COLUMNS = [
    'plan_tier',
    'monthly_fee',
    'total_spend_60d',
    'days_since_last_session',
    'sessions_recent_14d',
    'sessions_previous_14d',
    'session_drop_pct',
    'total_events_60d',
    'core_feature_usage_count',
    'avg_session_duration_minutes',
    'avg_session_gap_days',
    'support_ticket_count',
    'unresolved_tickets'
]

TARGET_COLUMN = 'churn_label'
EXCLUDE_COLUMNS = ['user_id']

def load_raw_feature_store(filepath="data/churn_feature_store.csv"):
    """Loads feature store CSV from data directory."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Feature store CSV not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def inspect_dataset_summary(df):
    """Outputs target distribution and missing value summary for inspection."""
    print("=" * 70)
    print("        CHURNGUARD AI - DAY 2 STEP 1 DATASET INSPECTION        ")
    print("=" * 70)
    
    total_rows = len(df)
    churn_counts = df[TARGET_COLUMN].value_counts()
    churn_pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    
    print(f"\n[1] DATASET DIMENSIONS:")
    print(f" - Total User Records: {total_rows}")
    print(f" - Total Columns:      {len(df.columns)}")

    print(f"\n[2] TARGET CLASS DISTRIBUTION ('{TARGET_COLUMN}'):")
    print(f" - Active Users (0):   {churn_counts.get(0, 0)} ({churn_pcts.get(0, 0.0):.2f}%)")
    print(f" - Churned Users (1):  {churn_counts.get(1, 0)} ({churn_pcts.get(1, 0.0):.2f}%)")
    print(f" - Class Imbalance:    ~1:{int(round(churn_counts.get(0,0) / max(1, churn_counts.get(1,0))))}")

    print(f"\n[3] MISSING VALUE SUMMARY:")
    missing_series = df[RAW_FEATURE_COLUMNS].isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    if missing_cols.empty:
        print(" - No missing values found across raw feature columns.")
    else:
        for col, cnt in missing_cols.items():
            pct = (cnt / total_rows) * 100
            print(f" - Column '{col}': {cnt} missing values ({pct:.2f}%)")

def prepare_train_test_data(filepath="data/churn_feature_store.csv", test_size=0.20, random_state=42):
    """
    Prepares train/test splits for both Logistic Regression (Scaled + Imputed)
    and XGBoost (One-Hot Encoded + Native Missing Values).
    """
    df = load_raw_feature_store(filepath)
    
    # 1. One-Hot Encoding for plan_tier (Creates plan_tier_Basic, plan_tier_Pro, plan_tier_Enterprise)
    df_encoded = pd.get_dummies(df, columns=['plan_tier'], dtype=int)
    
    # Build Feature Matrix X and Target vector y
    one_hot_cols = ['plan_tier_Basic', 'plan_tier_Pro', 'plan_tier_Enterprise']
    numeric_cols = [c for c in RAW_FEATURE_COLUMNS if c != 'plan_tier']
    all_feature_cols = numeric_cols + one_hot_cols
    
    X = df_encoded[all_feature_cols].copy()
    y = df_encoded[TARGET_COLUMN].copy()

    # 2. Stratified 80/20 Train/Test Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. XGBoost Data: Retains np.nan for missing gap values (Native tree handling)
    X_train_xgb = X_train_raw.copy()
    X_test_xgb = X_test_raw.copy()

    # 4. Logistic Regression Data: Median Imputation + StandardScaler
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_lr = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(X_train_raw)),
        columns=all_feature_cols, index=X_train_raw.index
    )
    X_test_lr = pd.DataFrame(
        scaler.transform(imputer.transform(X_test_raw)),
        columns=all_feature_cols, index=X_test_raw.index
    )

    return {
        'X_train_xgb': X_train_xgb, 'X_test_xgb': X_test_xgb,
        'X_train_lr': X_train_lr, 'X_test_lr': X_test_lr,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': all_feature_cols
    }

if __name__ == "__main__":
    df_raw = load_raw_feature_store()
    inspect_dataset_summary(df_raw)
    data_dict = prepare_train_test_data()
    print(f"\n[4] ONE-HOT ENCODED TRAIN / TEST SPLIT COMPLETED SUCCESSFULLY:")
    print(f" - Feature Columns ({len(data_dict['feature_names'])}): {data_dict['feature_names']}")
    print(f" - Training Set Shape:  X_train={data_dict['X_train_xgb'].shape}, y_train={data_dict['y_train'].shape}")
    print(f" - Testing Set Shape:   X_test={data_dict['X_test_xgb'].shape}, y_test={data_dict['y_test'].shape}")
    print("=" * 70)
