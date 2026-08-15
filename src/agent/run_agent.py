import os
import sys
import json

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.agent import RetentionAgent

def run_agent_test_suite():
    print("=" * 80)
    print("      CHURNGUARD AI - DAY 4 STEP 2 RETENTION AGENT TOOL-CALLING TEST      ")
    print("=" * 80)

    agent = RetentionAgent()

    # Load local SHAP explanations for User #3510 if available
    shap_json_path = "data/shap_local_explanations.json"
    user_3510_drivers = [
        {'feature': 'sessions_recent_14d', 'feature_value': 0.0, 'shap_value': 1.1199},
        {'feature': 'support_ticket_count', 'feature_value': 2.0, 'shap_value': 0.7479},
        {'feature': 'days_since_last_session', 'feature_value': 23.0, 'shap_value': 0.6897}
    ]

    if os.path.exists(shap_json_path):
        with open(shap_json_path, 'r') as f:
            local_db = json.load(f)
            u3510 = next((r for r in local_db if r['user_id'] == 3510), None)
            if u3510:
                user_3510_drivers = [u3510['top_driver_1'], u3510['top_driver_2'], u3510['top_driver_3']]

    test_cases = [
        {
            'test_name': "TEST 1: Example User #3510 (High-Risk Pro Account)",
            'user_profile': {
                'user_id': 3510,
                'churn_probability': 0.8351,
                'risk_tier': "High Risk",
                'plan_tier': "Pro",
                'monthly_fee': 99.00,
                'top_3_shap_drivers': user_3510_drivers
            }
        },
        {
            'test_name': "TEST 2: High-Risk / High-Value Enterprise Account (User #1005)",
            'user_profile': {
                'user_id': 1005,
                'churn_probability': 0.8800,
                'risk_tier': "High Risk",
                'plan_tier': "Enterprise",
                'monthly_fee': 299.00,
                'top_3_shap_drivers': [
                    {'feature': 'support_ticket_count', 'feature_value': 3.0, 'shap_value': 1.2500},
                    {'feature': 'unresolved_tickets', 'feature_value': 2.0, 'shap_value': 0.8500},
                    {'feature': 'days_since_last_session', 'feature_value': 14.0, 'shap_value': 0.4500}
                ]
            }
        },
        {
            'test_name': "TEST 3: Low-Risk Basic Account (User #1099)",
            'user_profile': {
                'user_id': 1099,
                'churn_probability': 0.1200,
                'risk_tier': "Low Risk",
                'plan_tier': "Basic",
                'monthly_fee': 29.00,
                'top_3_shap_drivers': [
                    {'feature': 'total_events_60d', 'feature_value': 210.0, 'shap_value': -0.8500},
                    {'feature': 'sessions_recent_14d', 'feature_value': 10.0, 'shap_value': -0.7500},
                    {'feature': 'days_since_last_session', 'feature_value': 2.0, 'shap_value': -0.5000}
                ]
            }
        },
        {
            'test_name': "TEST 4: Guardrail Enforcement Test (Excessive 30% Discount Request on Basic Plan)",
            'user_profile': {
                'user_id': 2040,
                'churn_probability': 0.7500,
                'risk_tier': "High Risk",
                'plan_tier': "Basic",
                'monthly_fee': 29.00,
                'override_proposed_discount_pct': 30.0,
                'override_action_name': "30% Discount Promotion",
                'top_3_shap_drivers': [
                    {'feature': 'days_since_last_session', 'feature_value': 18.0, 'shap_value': 0.9500},
                    {'feature': 'session_drop_pct', 'feature_value': 80.0, 'shap_value': 0.7000},
                    {'feature': 'support_ticket_count', 'feature_value': 1.0, 'shap_value': 0.3000}
                ]
            }
        }
    ]

    for tc in test_cases:
        print(f"\n===============================================================================")
        print(f" {tc['test_name']}")
        print(f"===============================================================================")
        
        prof = tc['user_profile']
        print(f"[INPUT RISK PROFILE]")
        print(f" - User ID:           {prof['user_id']}")
        print(f" - Churn Probability: {prof['churn_probability']*100:.2f}%")
        print(f" - Risk Tier:         {prof['risk_tier']}")
        print(f" - Plan Tier:         {prof['plan_tier']} (${prof['monthly_fee']}/mo)")
        print(f" - Primary SHAP Driver: {prof['top_3_shap_drivers'][0]['feature']} (SHAP: {prof['top_3_shap_drivers'][0]['shap_value']:+.4f})")

        # Execute RetentionAgent Workflow
        result = agent.process_user_risk_profile(prof)

        print(f"\n[AGENT STRATEGY & DECISION]")
        print(f" - Selected Strategy: {result['agent_strategy']}")
        print(f" - Execution Status:  {result['execution_status']}")

        print(f"\n[TOOLS CALLED TRACE ({len(result['tool_call_trace'])} Tool Calls)]")
        for idx, call in enumerate(result['tool_call_trace'], 1):
            t_name = call['tool']
            print(f"  {idx}. Tool: [{t_name}]")
            if t_name == 'check_retention_rules':
                print(f"     -> Check Iteration {call.get('iteration', 1)}: Approved = {call['output']['approved']}")
                print(f"     -> Reason: \"{call['output']['reason']}\"")
            elif t_name == 'calculate_intervention_roi':
                print(f"     -> Value Saved: ${call['output']['expected_value_saved']:.2f} | Cost: ${call['output']['intervention_cost']:.2f} | Net Impact: ${call['output']['net_impact']:.2f}")

        print(f"\n[GUARDRAIL RESULT]")
        print(f" - Guardrail Approved: {result['guardrail_approved']}")
        if not result['guardrail_approved']:
            print(f" - Rejection Reason:   \"{result['rejection_reason']}\"")

        if result['guardrail_approved']:
            roi = result['roi_simulation']
            msg = result['generated_message']
            print(f"\n[ROI RESULT]")
            print(f" - Calculated LTV:       ${result['calculated_ltv']:.2f}")
            print(f" - Applied Discount:     {result['applied_discount_pct']:.1f}%")
            print(f" - Expected Value Saved: ${roi['expected_value_saved']:.2f}")
            print(f" - Total Cost:           ${roi['intervention_cost']:.2f}")
            print(f" - Net Impact / ROI:     ${roi['net_impact']:.2f} (ROI: {roi['roi_percentage']}%)")

            print(f"\n[GENERATED RETENTION MESSAGE]")
            print(f" - Category:             [{msg['template_category']}]")
            print(f" - Subject Line:         \"{msg['subject_line']}\"")
            print(f" - Message Body:         \"{msg['message_body']}\"")

    print("\n" + "=" * 80)
    print("        ALL 4 RETENTION AGENT TEST CASES EXECUTED SUCCESSFULLY!        ")
    print("=" * 80)

if __name__ == "__main__":
    run_agent_test_suite()
