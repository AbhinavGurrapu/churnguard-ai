import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.tools import (
    calculate_ltv,
    check_retention_rules,
    calculate_intervention_roi
)

SIMULATION_CSV_PATH = "data/retention_impact_simulation.csv"
SIMULATION_JSON_PATH = "data/retention_impact_summary.json"

# Explicit Central Simulation Assumptions
SIMULATION_ASSUMPTIONS = {
    "expected_retention_probability": 0.30,  # Assumed baseline 30% conversion success rate for targeted retention plays
    "messaging_cost": 1.00,                  # $1.00 delivery/communication cost per targeted account
    "lifespan_months": 12,                   # 12-month standard customer LTV horizon
    "risk_threshold": 0.35,                  # Minimum churn probability cutoff for intervention eligibility
    "disclaimer": "SIMULATION ASSUMPTIONS: These financial metrics are expected-value projections based on probabilistic model predictions, not empirically observed causal field results."
}

def simulate_user_impact(user_row: pd.Series) -> dict:
    """
    Calculates deterministic business impact metrics for a single user record.
    Enforces risk eligibility, tier-based discount capping, and division-by-zero safety.
    """
    uid = int(user_row['user_id'])
    churn_prob = float(user_row['churn_probability'])
    monthly_fee = float(user_row['monthly_fee'])
    plan_tier = str(user_row['plan_tier'])
    primary_driver = str(user_row.get('driver_1_feature', 'sessions_recent_14d'))

    # 1. Calculate Authoritative Customer LTV
    ltv_data = calculate_ltv(monthly_fee, plan_tier, expected_lifespan_months=SIMULATION_ASSUMPTIONS["lifespan_months"])
    ltv = ltv_data['calculated_ltv']

    # 2. Determine Strategy & Initial Proposed Discount
    if primary_driver in ['sessions_recent_14d', 'days_since_last_session', 'session_drop_pct']:
        proposed_action = "Re-engagement Strategy Call"
        proposed_discount_pct = 15.0 if plan_tier != 'Enterprise' else 20.0
    elif primary_driver in ['support_ticket_count', 'unresolved_tickets']:
        proposed_action = "VIP Support Escalation"
        proposed_discount_pct = 15.0 if plan_tier != 'Enterprise' else 20.0
    else:
        proposed_action = "Renewal Value Review"
        proposed_discount_pct = 10.0

    # 3. Check Policy Guardrails
    rule_res = check_retention_rules(uid, churn_prob, ltv, proposed_action, proposed_discount_pct)
    
    # 4. Handle Eligibility & Guardrail Discount Capping
    if churn_prob < SIMULATION_ASSUMPTIONS["risk_threshold"]:
        # Low-risk user: Not targeted for intervention (prevents budget waste)
        is_eligible = False
        approved_action = "No Action (User Safe)"
        approved_discount_pct = 0.0
        expected_value_saved = 0.0
        discount_cost = 0.0
        total_cost = 0.0
        net_impact = 0.0
        roi_pct = 0.0
        status_note = "Ineligible: Churn risk below 35% threshold."
    else:
        # High-risk user: Target for intervention using policy-approved discount cap
        is_eligible = True
        approved_action = proposed_action
        approved_discount_pct = rule_res['max_allowed_discount_pct']
        
        # Calculate Intervention Financial ROI
        roi_data = calculate_intervention_roi(
            ltv=ltv,
            proposed_discount_pct=approved_discount_pct,
            est_retention_success_rate=SIMULATION_ASSUMPTIONS["expected_retention_probability"],
            messaging_cost=SIMULATION_ASSUMPTIONS["messaging_cost"]
        )
        
        expected_value_saved = roi_data['expected_value_saved']
        discount_cost = ltv * (approved_discount_pct / 100.0)
        total_cost = roi_data['intervention_cost']
        net_impact = roi_data['net_impact']
        
        # Division-by-zero protection for ROI calculation
        if total_cost > 0:
            roi_pct = (net_impact / total_cost) * 100.0
        else:
            roi_pct = 0.0
            
        status_note = rule_res['reason']

    return {
        'user_id': uid,
        'churn_probability': round(churn_prob, 4),
        'plan_tier': plan_tier,
        'monthly_fee': round(monthly_fee, 2),
        'ltv': round(ltv, 2),
        'primary_driver': primary_driver,
        'is_eligible': is_eligible,
        'approved_action': approved_action,
        'approved_discount_pct': round(approved_discount_pct, 1),
        'expected_value_saved': round(expected_value_saved, 2),
        'discount_cost': round(discount_cost, 2),
        'total_intervention_cost': round(total_cost, 2),
        'net_impact': round(net_impact, 2),
        'roi_percentage': round(roi_pct, 2),
        'status_note': status_note
    }

def run_portfolio_impact_simulation(
    feature_store_path: str = "data/churn_feature_store.csv",
    shap_explanations_path: str = "data/shap_local_explanations.csv"
) -> dict:
    """
    Runs portfolio-level retention impact simulation across all analyzed users.
    Generates user-level CSV breakdown and summary JSON metrics.
    """
    df_features = pd.read_csv(feature_store_path)
    
    # Merge SHAP local explanations if available, else load feature store
    if os.path.exists(shap_explanations_path):
        df_shap = pd.read_csv(shap_explanations_path)
        df_merged = pd.merge(
            df_shap[['user_id', 'churn_probability', 'driver_1_feature']],
            df_features[['user_id', 'plan_tier', 'monthly_fee', 'churn_label']],
            on='user_id',
            how='inner'
        )
    else:
        # Fallback to feature store with baseline churn probability
        df_merged = df_features.copy()
        df_merged['churn_probability'] = df_merged['churn_label'].astype(float)
        df_merged['driver_1_feature'] = 'sessions_recent_14d'

    # Run user-level simulation
    user_results = [simulate_user_impact(row) for _, row in df_merged.iterrows()]
    df_sim = pd.DataFrame(user_results)

    # Save detailed user-level simulation CSV
    os.makedirs(os.path.dirname(SIMULATION_CSV_PATH), exist_ok=True)
    df_sim.to_csv(SIMULATION_CSV_PATH, index=False)

    # Portfolio Aggregations
    total_users = len(df_sim)
    df_targeted = df_sim[df_sim['is_eligible'] == True]
    df_untargeted = df_sim[df_sim['is_eligible'] == False]

    eligible_high_risk_users = len(df_targeted)
    users_not_targeted = len(df_untargeted)

    total_ltv_targeted = float(df_targeted['ltv'].sum())
    total_expected_value_saved = float(df_targeted['expected_value_saved'].sum())
    total_intervention_cost = float(df_targeted['total_intervention_cost'].sum())
    total_net_impact = float(df_targeted['net_impact'].sum())

    # Average ROI % across targeted users (with division-by-zero protection)
    if total_intervention_cost > 0:
        overall_program_roi_pct = (total_net_impact / total_intervention_cost) * 100.0
    else:
        overall_program_roi_pct = 0.0

    positive_roi_users = int((df_targeted['net_impact'] > 0).sum())
    negative_roi_users = int((df_targeted['net_impact'] <= 0).sum())

    # Build Portfolio Summary Dictionary
    summary = {
        'simulation_assumptions': SIMULATION_ASSUMPTIONS,
        'portfolio_metrics': {
            'total_users_analyzed': total_users,
            'eligible_high_risk_users_targeted': eligible_high_risk_users,
            'users_not_targeted_safe': users_not_targeted,
            'total_ltv_of_targeted_users': round(total_ltv_targeted, 2),
            'total_expected_value_saved': round(total_expected_value_saved, 2),
            'total_intervention_cost': round(total_intervention_cost, 2),
            'total_net_impact': round(total_net_impact, 2),
            'overall_program_roi_percentage': round(overall_program_roi_pct, 2),
            'positive_roi_users_count': positive_roi_users,
            'negative_roi_users_count': negative_roi_users
        },
        'scenario_comparison': {
            'scenario_a_no_intervention': {
                'description': "Baseline: No retention program executed.",
                'users_targeted': 0,
                'expected_value_saved': 0.0,
                'intervention_cost': 0.0,
                'net_impact': 0.0
            },
            'scenario_b_retention_program': {
                'description': "Active: Target eligible high-risk users (churn prob >= 35%) with guardrail-capped discounts.",
                'users_targeted': eligible_high_risk_users,
                'expected_value_saved': round(total_expected_value_saved, 2),
                'intervention_cost': round(total_intervention_cost, 2),
                'net_impact': round(total_net_impact, 2),
                'net_uplift_over_baseline': round(total_net_impact, 2)
            }
        }
    }

    # Save portfolio summary JSON
    with open(SIMULATION_JSON_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
