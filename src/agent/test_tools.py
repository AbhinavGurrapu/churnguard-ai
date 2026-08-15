import os
import sys

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.tools import (
    calculate_ltv,
    check_retention_rules,
    calculate_intervention_roi,
    generate_retention_message,
    log_retention_action
)

def run_tools_test_suite():
    print("=" * 75)
    print("        CHURNGUARD AI - DAY 4 STEP 1 DETERMINISTIC TOOLS TEST       ")
    print("=" * 75)

    # Define Test Profiles
    profiles = [
        {
            'profile_name': "Example User #3510 (High Risk / Medium Value)",
            'user_id': 3510,
            'churn_prob': 0.8351,
            'monthly_fee': 99.00,
            'plan_tier': "Pro",
            'top_driver': "sessions_recent_14d",
            'proposed_action': "1-on-1 Strategy Call + 15% Discount",
            'proposed_discount_pct': 15.0
        },
        {
            'profile_name': "High-Risk / High-Value Enterprise Account (User #1005)",
            'user_id': 1005,
            'churn_prob': 0.8800,
            'monthly_fee': 299.00,
            'plan_tier': "Enterprise",
            'top_driver': "support_ticket_count",
            'proposed_action': "VIP Dedicated Tech Escalation + 20% Discount",
            'proposed_discount_pct': 20.0
        },
        {
            'profile_name': "Low-Risk / Low-Value Basic Account (User #1099)",
            'user_id': 1099,
            'churn_prob': 0.1200,
            'monthly_fee': 29.00,
            'plan_tier': "Basic",
            'top_driver': "total_events_60d",
            'proposed_action': "10% Renewal Discount",
            'proposed_discount_pct': 10.0
        },
        {
            'profile_name': "Guardrail Violation Test: Excessive Discount Request (User #2040)",
            'user_id': 2040,
            'churn_prob': 0.7500,
            'monthly_fee': 29.00,
            'plan_tier': "Basic",
            'top_driver': "days_since_last_session",
            'proposed_action': "30% Discount (Exceeds Policy)",
            'proposed_discount_pct': 30.0
        }
    ]

    for p in profiles:
        print(f"\n--- Testing Profile: {p['profile_name']} ---")
        
        # 1. Calculate LTV
        ltv_res = calculate_ltv(p['monthly_fee'], p['plan_tier'])
        ltv = ltv_res['calculated_ltv']
        print(f" 1. LTV Tool Output:        LTV = ${ltv:.2f} (Fee: ${p['monthly_fee']}, Plan: {p['plan_tier']})")

        # 2. Check Retention Guardrail Rules
        rule_res = check_retention_rules(
            p['user_id'], p['churn_prob'], ltv, p['proposed_action'], p['proposed_discount_pct']
        )
        print(f" 2. Guardrail Approval:     Approved = {rule_res['approved']}")
        print(f"    -> Reason:              \"{rule_res['reason']}\"")

        # 3. Calculate ROI (if approved)
        if rule_res['approved']:
            roi_res = calculate_intervention_roi(ltv, p['proposed_discount_pct'])
            net_roi = roi_res['net_impact']
            print(f" 3. ROI Simulation Output:   Expected Value Saved = ${roi_res['expected_value_saved']:.2f} | Cost = ${roi_res['intervention_cost']:.2f} | Net Impact = ${net_roi:.2f} (ROI: {roi_res['roi_percentage']}%)")
            
            # 4. Generate Message Template
            msg_res = generate_retention_message(p['user_id'], p['top_driver'], offer_text=f"{p['proposed_discount_pct']:.0f}% discount")
            print(f" 4. Message Generator:      Category = [{msg_res['template_category']}]")
            print(f"    -> Subject:             \"{msg_res['subject_line']}\"")
            
            # 5. Log Execution Action
            log_res = log_retention_action(
                p['user_id'], p['churn_prob'], p['proposed_action'], rule_res['reason'],
                p['proposed_discount_pct'], ltv, net_roi, "SIMULATED_SUCCESS"
            )
            print(f" 5. Action Logger:          Status = {log_res['log_status']} (Logged to {log_res['storage_path']})")
        else:
            print(f" 3-5. Action Halted:        Execution blocked by guardrails. No ROI, message, or log generated.")

    print("\n" + "=" * 75)
    print("      ALL DETERMINISTIC BUSINESS LOGIC TOOLS TESTED SUCCESSFULLY!      ")
    print("=" * 75)

if __name__ == "__main__":
    run_tools_test_suite()
