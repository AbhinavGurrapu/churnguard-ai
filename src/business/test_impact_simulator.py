import os
import sys
import pandas as pd
import json

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.business.impact_simulator import (
    simulate_user_impact,
    run_portfolio_impact_simulation,
    SIMULATION_ASSUMPTIONS
)

def run_impact_simulator_test_suite():
    print("=" * 80)
    print("      CHURNGUARD AI - DAY 5 BUSINESS IMPACT SIMULATOR TEST SUITE       ")
    print("=" * 80)

    # 1. User-Level Unit Test Cases
    test_cases = [
        {
            'case_name': "Case A: High-Risk Pro User (User #3510)",
            'row': pd.Series({'user_id': 3510, 'churn_probability': 0.8351, 'plan_tier': 'Pro', 'monthly_fee': 99.00, 'driver_1_feature': 'sessions_recent_14d'}),
            'expected_eligible': True,
            'expected_discount': 15.0
        },
        {
            'case_name': "Case B: High-Risk Enterprise User (User #1005)",
            'row': pd.Series({'user_id': 1005, 'churn_probability': 0.8800, 'plan_tier': 'Enterprise', 'monthly_fee': 299.00, 'driver_1_feature': 'support_ticket_count'}),
            'expected_eligible': True,
            'expected_discount': 20.0
        },
        {
            'case_name': "Case C: Low-Risk User (User #1099 - Not Targeted)",
            'row': pd.Series({'user_id': 1099, 'churn_probability': 0.1200, 'plan_tier': 'Basic', 'monthly_fee': 29.00, 'driver_1_feature': 'total_events_60d'}),
            'expected_eligible': False,
            'expected_discount': 0.0
        },
        {
            'case_name': "Case D: High-Risk Basic User (User #2040 - Capped at 10%)",
            'row': pd.Series({'user_id': 2040, 'churn_probability': 0.7500, 'plan_tier': 'Basic', 'monthly_fee': 29.00, 'driver_1_feature': 'days_since_last_session'}),
            'expected_eligible': True,
            'expected_discount': 10.0
        },
        {
            'case_name': "Case E: Zero-Cost / Edge Case (Division-by-Zero Safety)",
            'row': pd.Series({'user_id': 9999, 'churn_probability': 0.1000, 'plan_tier': 'Basic', 'monthly_fee': 0.00, 'driver_1_feature': 'sessions_recent_14d'}),
            'expected_eligible': False,
            'expected_discount': 0.0
        }
    ]

    all_passed = True
    print("\n[PART 1: USER-LEVEL SIMULATION UNIT TESTS]")
    for tc in test_cases:
        res = simulate_user_impact(tc['row'])
        is_pass = (res['is_eligible'] == tc['expected_eligible']) and (res['approved_discount_pct'] == tc['expected_discount'])
        if not is_pass:
            all_passed = False
            
        print(f"\n --- {tc['case_name']} ---")
        print(f"  - User ID:                {res['user_id']}")
        print(f"  - Churn Probability:      {res['churn_probability']*100:.1f}%")
        print(f"  - LTV:                    ${res['ltv']:.2f}")
        print(f"  - Intervention Eligible:   {res['is_eligible']} (Expected: {tc['expected_eligible']})")
        print(f"  - Approved Discount:      {res['approved_discount_pct']}% (Expected: {tc['expected_discount']}%)")
        print(f"  - Expected Value Saved:  ${res['expected_value_saved']:.2f}")
        print(f"  - Total Cost:            ${res['total_intervention_cost']:.2f}")
        print(f"  - Net Financial Impact:  ${res['net_impact']:.2f} (ROI: {res['roi_percentage']}%)")
        print(f"  - Test Status:            {'PASS' if is_pass else 'FAIL'}")

    # 2. Portfolio-Level Simulation Run
    print("\n[PART 2: RUNNING PORTFOLIO-LEVEL IMPACT SIMULATION (600 TEST USERS)]")
    summary = run_portfolio_impact_simulation()
    
    pm = summary['portfolio_metrics']
    sc_a = summary['scenario_comparison']['scenario_a_no_intervention']
    sc_b = summary['scenario_comparison']['scenario_b_retention_program']

    # 3. Formatted Final Business Report
    print("\n" + "=" * 80)
    print("CHURNGUARD AI — DAY 5 RETENTION IMPACT SIMULATION")
    print("=" * 80)
    print(f"Users analyzed:            {pm['total_users_analyzed']}")
    print(f"Eligible for intervention: {pm['eligible_high_risk_users_targeted']}")
    print(f"Total LTV targeted:        ${pm['total_ltv_of_targeted_users']:,.2f}")
    print(f"Expected value saved:     ${pm['total_expected_value_saved']:,.2f}")
    print(f"Intervention cost:        ${pm['total_intervention_cost']:,.2f}")
    print(f"Net financial impact:     ${pm['total_net_impact']:,.2f}")
    print(f"Average ROI:               {pm['overall_program_roi_percentage']:.2f}%")
    print(f"Positive ROI Users:        {pm['positive_roi_users_count']}")
    print(f"Negative ROI Users:        {pm['negative_roi_users_count']}")
    print("-" * 80)
    print("SCENARIO COMPARISON")
    print("-" * 80)
    print("No Intervention:")
    print(f"Expected value saved:     ${sc_a['expected_value_saved']:,.2f}")
    print(f"Cost:                     ${sc_a['intervention_cost']:,.2f}")
    print(f"Net impact:               ${sc_a['net_impact']:,.2f}")
    print("\nRetention Program:")
    print(f"Expected value saved:     ${sc_b['expected_value_saved']:,.2f}")
    print(f"Cost:                     ${sc_b['intervention_cost']:,.2f}")
    print(f"Net impact:               ${sc_b['net_impact']:,.2f}")
    print("=" * 80)

    print(f"\nALL DAY 5 TEST CASES PASSED: {all_passed}")
    print("=" * 80)

if __name__ == "__main__":
    run_impact_simulator_test_suite()
