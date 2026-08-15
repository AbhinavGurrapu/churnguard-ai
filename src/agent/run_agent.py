import os
import sys
import json
import time

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.agent import RetentionAgent

def run_real_gemini_test_suite():
    print("=" * 85)
    print("      CHURNGUARD AI - DAY 4 STEP 3 REAL GEMINI LLM TOOL-CALLING TEST       ")
    print("      Model: gemini-3.5-flash-lite | SDK: google-genai                   ")
    print("=" * 85)

    agent = RetentionAgent(model_name="gemini-3.5-flash-lite")

    # Load local SHAP drivers for User #3510 if available
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
            'test_name': "TEST 1: Example User #3510 (High-Risk Pro Account - Real Gemini Call)",
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
            'test_name': "TEST 2: Low-Risk Basic Account (User #1099 - Guardrail Rejection)",
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
            'test_name': "TEST 3: Guardrail Enforcement & Bounded Retry (User #2040 - Excessive 30% Discount)",
            'user_profile': {
                'user_id': 2040,
                'churn_probability': 0.7500,
                'risk_tier': "High Risk",
                'plan_tier': "Basic",
                'monthly_fee': 29.00,
                'override_proposed_discount_pct': 30.0,
                'override_action_name': "30% Excessive Discount Promo",
                'top_3_shap_drivers': [
                    {'feature': 'days_since_last_session', 'feature_value': 18.0, 'shap_value': 0.9500},
                    {'feature': 'session_drop_pct', 'feature_value': 80.0, 'shap_value': 0.7000},
                    {'feature': 'support_ticket_count', 'feature_value': 1.0, 'shap_value': 0.3000}
                ]
            }
        }
    ]

    for tc_idx, tc in enumerate(test_cases, 1):
        if tc_idx > 1:
            print(f"\n [Free-Tier Quota Pacing] Waiting 15s between test cases to prevent 429 rate limit...")
            time.sleep(15)

        print(f"\n===================================================================================")
        print(f" {tc['test_name']}")
        print(f"===================================================================================")
        
        prof = tc['user_profile']
        print(f"[INPUT RISK PROFILE]")
        print(f" - User ID:           {prof['user_id']}")
        print(f" - Churn Probability: {prof['churn_probability']*100:.2f}%")
        print(f" - Risk Tier:         {prof['risk_tier']}")
        print(f" - Plan Tier:         {prof['plan_tier']} (${prof['monthly_fee']}/mo)")
        print(f" - Primary SHAP Driver: {prof['top_3_shap_drivers'][0]['feature']} (SHAP: {prof['top_3_shap_drivers'][0]['shap_value']:+.4f})")

        # Execute REAL Gemini LLM RetentionAgent Workflow
        result = agent.process_user_risk_profile(prof)

        print(f"\n[GEMINI REASONING SUMMARY]")
        print(f" \"{result['gemini_reasoning_summary'].strip()}\"")

        print(f"\n[REAL GEMINI TOOL-CALLING TRACE ({len(result['tool_call_trace'])} Tool Calls Executed)]")
        for idx, call in enumerate(result['tool_call_trace'], 1):
            t_name = call['tool']
            print(f"  {idx}. [Python Tool]: {t_name}")
            print(f"     -> Input:  {call['input']}")
            if t_name == 'check_retention_rules':
                print(f"     -> Guardrail Check Iteration {call.get('iteration', 1)}: Approved = {call['output']['approved']}")
                print(f"     -> Policy Response: \"{call['output']['reason']}\"")
            elif t_name == 'calculate_intervention_roi':
                print(f"     -> Value Saved: ${call['output']['expected_value_saved']:.2f} | Cost: ${call['output']['intervention_cost']:.2f} | Net Impact: ${call['output']['net_impact']:.2f}")

        print(f"\n[GUARDRAIL & EXECUTION RESULT]")
        print(f" - Guardrail Approved: {result['guardrail_approved']}")
        print(f" - Policy Reason:       \"{result['guardrail_reason']}\"")
        print(f" - Execution Status:    {result['execution_status']}")

        if result['guardrail_approved'] and result['roi_simulation']:
            roi = result['roi_simulation']
            msg = result['generated_message']
            print(f"\n[ROI SIMULATION RESULT]")
            print(f" - Calculated LTV:       ${result['calculated_ltv']:.2f}")
            print(f" - Expected Value Saved: ${roi['expected_value_saved']:.2f}")
            print(f" - Total Cost:           ${roi['intervention_cost']:.2f}")
            print(f" - Net Impact / ROI:     ${roi['net_impact']:.2f} (ROI: {roi['roi_percentage']}%)")

            if msg:
                print(f"\n[GENERATED RETENTION MESSAGE]")
                print(f" - Subject Line:         \"{msg['subject_line']}\"")
                print(f" - Message Body:         \"{msg['message_body']}\"")

    print("\n" + "=" * 85)
    print("      REAL GEMINI LLM TOOL-CALLING TEST SUITE COMPLETED SUCCESSFULLY!       ")
    print("=" * 85)

if __name__ == "__main__":
    run_real_gemini_test_suite()
