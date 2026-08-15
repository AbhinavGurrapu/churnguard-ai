import os
import sys
import json
import logging
from typing import Dict, Any, List

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.tools import (
    calculate_ltv,
    check_retention_rules,
    calculate_intervention_roi,
    generate_retention_message,
    log_retention_action
)

class RetentionAgent:
    """
    Single Autonomous Retention Agent for ChurnGuard AI.
    Orchestrates LLM strategic reasoning with deterministic Python business tools.
    """
    def __init__(self, api_key: str = None):
        # API credentials check (reads from env if not provided)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.max_tool_iterations = 5
        
    def process_user_risk_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the retention workflow for a single user risk profile.
        Enforces strict boundary between LLM strategic recommendation and deterministic Python guardrails.
        """
        uid = user_profile['user_id']
        churn_prob = float(user_profile['churn_probability'])
        plan_tier = user_profile['plan_tier']
        monthly_fee = float(user_profile['monthly_fee'])
        top_drivers = user_profile.get('top_3_shap_drivers', [])
        
        tool_call_trace: List[Dict[str, Any]] = []

        # 1. Step 1: Calculate Deterministic LTV
        ltv_res = calculate_ltv(monthly_fee, plan_tier)
        tool_call_trace.append({'tool': 'calculate_ltv', 'input': {'monthly_fee': monthly_fee, 'plan_tier': plan_tier}, 'output': ltv_res})
        ltv = ltv_res['calculated_ltv']

        # 2. Step 2: Determine Strategic Action & Proposed Discount
        primary_driver_feature = top_drivers[0]['feature'] if top_drivers else 'sessions_recent_14d'
        
        # Initial Proposed Strategy based on SHAP primary risk vector
        if primary_driver_feature in ['sessions_recent_14d', 'days_since_last_session', 'session_drop_pct']:
            strategy_name = "1-on-1 Product Onboarding & Re-engagement"
            initial_discount_pct = 15.0 if plan_tier != 'Enterprise' else 20.0
        elif primary_driver_feature in ['support_ticket_count', 'unresolved_tickets']:
            strategy_name = "VIP Dedicated Technical Escalation"
            initial_discount_pct = 15.0 if plan_tier != 'Enterprise' else 20.0
        else:
            strategy_name = "Account Executive Renewal Review"
            initial_discount_pct = 10.0

        # Handle explicit test override if passed (e.g. deliberate guardrail violation testing)
        if user_profile.get('override_proposed_discount_pct') is not None:
            initial_discount_pct = float(user_profile['override_proposed_discount_pct'])
            strategy_name = user_profile.get('override_action_name', strategy_name)

        # 3. Step 3: Check Deterministic Guardrails (Iteration 1)
        rule_res = check_retention_rules(
            user_id=uid,
            churn_prob=churn_prob,
            ltv=ltv,
            proposed_action=strategy_name,
            proposed_discount_pct=initial_discount_pct
        )
        tool_call_trace.append({'tool': 'check_retention_rules', 'iteration': 1, 'input': {'action': strategy_name, 'discount_pct': initial_discount_pct}, 'output': rule_res})

        # 4. Step 4: Handle Guardrail Rejection & Compliant Retry (Max 1 Retry)
        if not rule_res['approved']:
            # If rejected due to risk eligibility cutoff (<35% churn prob), halt execution cleanly
            if churn_prob < 0.35:
                return {
                    'user_id': uid,
                    'churn_probability': churn_prob,
                    'risk_tier': user_profile.get('risk_tier', 'Low'),
                    'agent_strategy': "No Action (User Safe)",
                    'guardrail_approved': False,
                    'rejection_reason': rule_res['reason'],
                    'tool_call_trace': tool_call_trace,
                    'execution_status': "HALTED_SAFE_USER"
                }

            # If rejected due to excessive discount, apply ONE compliant retry capped at max allowed discount
            max_allowed = rule_res['max_allowed_discount_pct']
            retry_discount_pct = max_allowed
            
            retry_rule_res = check_retention_rules(
                user_id=uid,
                churn_prob=churn_prob,
                ltv=ltv,
                proposed_action=strategy_name,
                proposed_discount_pct=retry_discount_pct
            )
            tool_call_trace.append({'tool': 'check_retention_rules', 'iteration': 2, 'input': {'action': strategy_name, 'discount_pct': retry_discount_pct}, 'output': retry_rule_res})

            if not retry_rule_res['approved']:
                return {
                    'user_id': uid,
                    'churn_probability': churn_prob,
                    'risk_tier': user_profile.get('risk_tier', 'High'),
                    'agent_strategy': strategy_name,
                    'guardrail_approved': False,
                    'rejection_reason': retry_rule_res['reason'],
                    'tool_call_trace': tool_call_trace,
                    'execution_status': "HALTED_GUARDRAIL_REJECTED"
                }
            
            # Use compliant retry parameters
            rule_res = retry_rule_res
            initial_discount_pct = retry_discount_pct

        # 5. Step 5: Execute Approved Financial ROI Simulation
        roi_res = calculate_intervention_roi(
            ltv=ltv,
            proposed_discount_pct=initial_discount_pct,
            est_retention_success_rate=0.30
        )
        tool_call_trace.append({'tool': 'calculate_intervention_roi', 'input': {'ltv': ltv, 'discount_pct': initial_discount_pct}, 'output': roi_res})

        # 6. Step 6: Generate Deterministic Personalized Message
        offer_text = f"{initial_discount_pct:.0f}% subscription discount" if initial_discount_pct > 0 else None
        msg_res = generate_retention_message(
            user_id=uid,
            top_driver_feature=primary_driver_feature,
            user_name=f"Customer #{uid}",
            offer_text=offer_text
        )
        tool_call_trace.append({'tool': 'generate_retention_message', 'input': {'primary_driver': primary_driver_feature}, 'output': msg_res})

        # 7. Step 7: Log Execution Record
        log_res = log_retention_action(
            user_id=uid,
            churn_prob=churn_prob,
            action_name=strategy_name,
            reason=rule_res['reason'],
            discount_pct=initial_discount_pct,
            ltv=ltv,
            net_roi=roi_res['net_impact'],
            status="EXECUTED_SUCCESS"
        )
        tool_call_trace.append({'tool': 'log_retention_action', 'input': {'action': strategy_name, 'net_roi': roi_res['net_impact']}, 'output': log_res})

        return {
            'user_id': uid,
            'churn_probability': churn_prob,
            'risk_tier': user_profile.get('risk_tier', 'High'),
            'plan_tier': plan_tier,
            'calculated_ltv': ltv,
            'agent_strategy': strategy_name,
            'applied_discount_pct': initial_discount_pct,
            'guardrail_approved': True,
            'guardrail_reason': rule_res['reason'],
            'roi_simulation': roi_res,
            'generated_message': msg_res,
            'tool_call_trace': tool_call_trace,
            'execution_status': "EXECUTED_SUCCESS"
        }
