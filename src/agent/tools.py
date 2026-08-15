import os
import json
import csv
from datetime import datetime

ACTION_LOG_PATH = "data/retention_action_logs.csv"

# 1. TOOL 1: calculate_ltv()
def calculate_ltv(monthly_fee, plan_tier="Basic", expected_lifespan_months=12):
    """
    Simplified SaaS Customer Lifetime Value (LTV) calculation.
    Assumption: Average customer tenure is estimated at 12 months for this user cohort.
    LTV = Monthly Fee * Expected Lifespan Months
    """
    ltv = float(monthly_fee) * expected_lifespan_months
    return {
        'monthly_fee': float(monthly_fee),
        'plan_tier': plan_tier,
        'expected_lifespan_months': expected_lifespan_months,
        'calculated_ltv': round(ltv, 2),
        'assumption_note': "LTV calculated assuming a baseline 12-month customer retention lifespan."
    }

# 2. TOOL 2: check_retention_rules()
def check_retention_rules(user_id, churn_prob, ltv, proposed_action, proposed_discount_pct=0.0):
    """
    Deterministic business guardrail validator.
    Enforces risk eligibility, maximum discount caps by LTV tier, and business constraints.
    Returns approved = True/False and detailed validation status.
    """
    churn_prob = float(churn_prob)
    ltv = float(ltv)
    proposed_discount_pct = float(proposed_discount_pct)
    
    # Guardrail Rule 1: Risk Eligibility Cutoff
    if churn_prob < 0.35:
        return {
            'user_id': user_id,
            'approved': False,
            'reason': f"User churn probability ({churn_prob*100:.1f}%) is below the 35% risk threshold. Ineligible for retention discount.",
            'max_allowed_discount_pct': 0.0,
            'proposed_discount_pct': proposed_discount_pct
        }

    # Guardrail Rule 2: Tiered Maximum Discount Caps by LTV
    if ltv >= 2500.0:
        max_discount = 20.0  # High LTV / Enterprise
    elif ltv >= 800.0:
        max_discount = 15.0  # Medium LTV / Pro
    else:
        max_discount = 10.0  # Low LTV / Basic

    # Guardrail Rule 3: Discount Limit Validation
    if proposed_discount_pct > max_discount:
        return {
            'user_id': user_id,
            'approved': False,
            'reason': f"Proposed discount ({proposed_discount_pct:.1f}%) exceeds maximum allowed cap of {max_discount:.1f}% for LTV ${ltv:.2f}.",
            'max_allowed_discount_pct': max_discount,
            'proposed_discount_pct': proposed_discount_pct
        }

    return {
        'user_id': user_id,
        'approved': True,
        'reason': f"Action '{proposed_action}' approved. Proposed discount ({proposed_discount_pct:.1f}%) within policy cap of {max_discount:.1f}%.",
        'max_allowed_discount_pct': max_discount,
        'proposed_discount_pct': proposed_discount_pct
    }

# 3. TOOL 3: calculate_intervention_roi()
def calculate_intervention_roi(ltv, proposed_discount_pct, est_retention_success_rate=0.30, messaging_cost=1.00):
    """
    Estimates financial ROI of a retention intervention.
    Expected Value Saved = LTV * Estimated Retention Success Rate
    Intervention Cost = (LTV * Discount %) + Messaging Cost
    Net Impact = Expected Value Saved - Intervention Cost
    ROI % = (Net Impact / Intervention Cost) * 100
    """
    ltv = float(ltv)
    discount_pct = float(proposed_discount_pct)
    
    expected_value_saved = ltv * est_retention_success_rate
    discount_cost = ltv * (discount_pct / 100.0)
    total_cost = discount_cost + messaging_cost
    net_impact = expected_value_saved - total_cost
    roi_pct = (net_impact / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        'ltv': round(ltv, 2),
        'proposed_discount_pct': discount_pct,
        'estimated_success_rate': est_retention_success_rate,
        'expected_value_saved': round(expected_value_saved, 2),
        'intervention_cost': round(total_cost, 2),
        'net_impact': round(net_impact, 2),
        'roi_percentage': round(roi_pct, 2),
        'assumption_note': "Simulation estimate assuming a 30% baseline retention conversion rate."
    }

# 4. TOOL 4: generate_retention_message()
def generate_retention_message(user_id, top_driver_feature, user_name="Valued Customer", offer_text=None):
    """
    Deterministic message template generator based on top SHAP driver.
    """
    offer_str = f" As a special gesture, we're offering: {offer_text}." if offer_text else ""
    
    if top_driver_feature in ['sessions_recent_14d', 'days_since_last_session', 'session_drop_pct']:
        headline = "We Miss You at ChurnGuard!"
        body = f"Hi {user_name}, we noticed your team hasn't logged in recently. We've reserved a complimentary 1-on-1 strategy call with our product specialist to help you get back on track.{offer_str}"
        category = "Re-engagement"
        
    elif top_driver_feature in ['support_ticket_count', 'unresolved_tickets']:
        headline = "Dedicated Support Update for Your Account"
        body = f"Hi {user_name}, we saw you recently experienced technical support issues. Our senior engineering team has escalated your open items for immediate resolution.{offer_str}"
        category = "Friction Resolution"
        
    elif top_driver_feature in ['core_feature_usage_count', 'total_events_60d']:
        headline = "Unlock Full Value from Your Subscription"
        body = f"Hi {user_name}, discover advanced workflows and automation features in your dashboard to supercharge your team's results.{offer_str}"
        category = "Feature Adoption"
        
    else:
        headline = "Thank You for Being a Valued Partner"
        body = f"Hi {user_name}, we appreciate your continued partnership with ChurnGuard AI. Let us know how we can better support your business goals.{offer_str}"
        category = "General Appreciation"

    return {
        'user_id': user_id,
        'template_category': category,
        'subject_line': headline,
        'message_body': body
    }

# 5. TOOL 5: log_retention_action()
def log_retention_action(user_id, churn_prob, action_name, reason, discount_pct, ltv, net_roi, status):
    """
    Simulates local execution logging by appending structured execution records to CSV.
    """
    os.makedirs(os.path.dirname(ACTION_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(ACTION_LOG_PATH)
    
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_entry = {
        'timestamp': timestamp_str,
        'user_id': user_id,
        'churn_probability': round(float(churn_prob), 4),
        'action_name': action_name,
        'reason': reason,
        'discount_pct': round(float(discount_pct), 1),
        'ltv': round(float(ltv), 2),
        'net_roi': round(float(net_roi), 2),
        'execution_status': status
    }
    
    fieldnames = ['timestamp', 'user_id', 'churn_probability', 'action_name', 'reason', 'discount_pct', 'ltv', 'net_roi', 'execution_status']
    
    with open(ACTION_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)
        
    return {
        'log_status': "SUCCESS",
        'logged_entry': log_entry,
        'storage_path': ACTION_LOG_PATH
    }
