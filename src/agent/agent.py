import os
import sys
import json
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# Ensure project root is in Python path
sys.path.append(os.getcwd())

from src.agent.tools import (
    calculate_ltv,
    check_retention_rules,
    calculate_intervention_roi,
    generate_retention_message,
    log_retention_action
)

# Helper function to load .env safely without third-party dependencies
def _load_env_file():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

class RetentionAgent:
    """
    Autonomous Retention Agent for ChurnGuard AI using REAL Google Gemini LLM Tool-Calling.
    Connects Gemini (gemini-3.5-flash) with deterministic Python business tools.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash-lite"):
        _load_env_file()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing. Please configure it in your environment or .env file.")
        
        self.model_name = model_name
        self.client = genai.Client()

    def process_user_risk_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates a REAL Gemini LLM tool-calling session for a single user risk profile.
        Gemini handles contextual reasoning, strategy selection, and tool calls.
        Python tools enforce LTV, ROI, discount caps, policy guardrails, and logging.
        """
        uid = int(user_profile['user_id'])
        churn_prob = float(user_profile['churn_probability'])
        plan_tier = str(user_profile['plan_tier'])
        monthly_fee = float(user_profile['monthly_fee'])
        top_drivers = user_profile.get('top_3_shap_drivers', [])

        tool_call_trace: List[Dict[str, Any]] = []

        # Define wrapped tools that record tool call traces transparently
        def tool_calculate_ltv(monthly_fee_val: float, plan_tier_val: str) -> str:
            """Calculates annual customer Lifetime Value (LTV)."""
            res = calculate_ltv(monthly_fee_val, plan_tier_val)
            tool_call_trace.append({'tool': 'calculate_ltv', 'input': {'monthly_fee': monthly_fee_val, 'plan_tier': plan_tier_val}, 'output': res})
            return json.dumps(res)

        def tool_check_retention_rules(user_id_val: int, churn_prob_val: float, ltv_val: float, proposed_action_val: str, proposed_discount_pct_val: float) -> str:
            """Validates proposed retention action and discount percentage against business guardrail policies."""
            res = check_retention_rules(user_id_val, churn_prob_val, ltv_val, proposed_action_val, proposed_discount_pct_val)
            iteration_count = len([t for t in tool_call_trace if t['tool'] == 'check_retention_rules']) + 1
            tool_call_trace.append({'tool': 'check_retention_rules', 'iteration': iteration_count, 'input': {'user_id': user_id_val, 'action': proposed_action_val, 'discount_pct': proposed_discount_pct_val}, 'output': res})
            return json.dumps(res)

        def tool_calculate_intervention_roi(ltv_val: float, proposed_discount_pct_val: float) -> str:
            """Calculates financial expected value saved, intervention cost, net impact, and ROI percentage."""
            res = calculate_intervention_roi(ltv_val, proposed_discount_pct_val)
            tool_call_trace.append({'tool': 'calculate_intervention_roi', 'input': {'ltv': ltv_val, 'discount_pct': proposed_discount_pct_val}, 'output': res})
            return json.dumps(res)

        def tool_generate_retention_message(user_id_val: int, top_driver_feature_val: str, user_name_val: str = "Valued Customer", offer_text_val: str = None) -> str:
            """Generates personalized retention message template based on primary SHAP risk vector."""
            res = generate_retention_message(user_id_val, top_driver_feature_val, user_name_val, offer_text_val)
            tool_call_trace.append({'tool': 'generate_retention_message', 'input': {'user_id': user_id_val, 'top_driver': top_driver_feature_val}, 'output': res})
            return json.dumps(res)

        def tool_log_retention_action(user_id_val: int, churn_prob_val: float, action_name_val: str, reason_val: str, discount_pct_val: float, ltv_val: float, net_roi_val: float, status_val: str) -> str:
            """Logs final retention execution action to local structured storage."""
            res = log_retention_action(user_id_val, churn_prob_val, action_name_val, reason_val, discount_pct_val, ltv_val, net_roi_val, status_val)
            tool_call_trace.append({'tool': 'log_retention_action', 'input': {'user_id': user_id_val, 'action': action_name_val, 'status': status_val}, 'output': res})
            return json.dumps(res)

        tools_list = [
            tool_calculate_ltv,
            tool_check_retention_rules,
            tool_calculate_intervention_roi,
            tool_generate_retention_message,
            tool_log_retention_action
        ]

        system_instruction = """
        You are the Autonomous Retention Agent for ChurnGuard AI.
        Your job is to analyze customer churn risk profiles, select an optimal retention strategy based on SHAP drivers, and call Python tools via function calling.

        MANDATORY TOOL OPERATIONAL INSTRUCTIONS:
        1. Call `tool_calculate_ltv` first to get authoritative LTV.
        2. Propose a retention action and discount percentage based on SHAP risk drivers.
        3. Call `tool_check_retention_rules` to validate your proposed action.
        4. IF `tool_check_retention_rules` returns approved = False:
           - If churn probability is < 0.35, DO NOT offer any discount. Halt execution cleanly.
           - If discount exceeds allowed cap, make AT MOST ONE RETRY with discount_pct set to the max_allowed_discount_pct returned by the tool.
           - If retry is also rejected, stop safely. Do NOT make further retries.
        5. IF approved = True:
           - Call `tool_calculate_intervention_roi` to get exact ROI math.
           - Call `tool_generate_retention_message` to construct customer copy.
           - Call `tool_log_retention_action` to record execution with status "EXECUTED_SUCCESS".

        NEVER invent LTV or ROI numbers. Rely strictly on Python tool returns.
        """

        prompt = f"""
        Analyze the following Customer Churn Risk Profile and execute the appropriate retention workflow using function tools:

        Customer Profile:
        - User ID: {uid}
        - Churn Probability: {churn_prob:.4f} ({churn_prob*100:.1f}%)
        - Risk Tier: {user_profile.get('risk_tier', 'Unknown')}
        - Plan Tier: {plan_tier}
        - Monthly Fee: ${monthly_fee:.2f}
        - Top SHAP Drivers: {json.dumps(top_drivers, indent=2)}

        Execute the tool sequence now.
        """

        # Handle explicit override test case if provided (e.g. deliberate excessive discount test)
        if user_profile.get('override_proposed_discount_pct') is not None:
            override_discount = float(user_profile['override_proposed_discount_pct'])
            prompt += f"\nNote for Strategy Testing: Please initially attempt a proposed discount of {override_discount:.1f}% for action '{user_profile.get('override_action_name', 'Discount Promotion')}' to test policy guardrail validation."

        # Create Gemini Chat Session with Automatic Function Calling (AFC)
        chat = self.client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools_list,
                temperature=0.1
            )
        )

        # Retry loop to handle API rate limits (429) and temporary high demand (503) gracefully
        max_retries = 5
        response = None
        for attempt in range(max_retries):
            try:
                response = chat.send_message(prompt)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str or "UNAVAILABLE" in err_str:
                    if attempt < max_retries - 1:
                        wait_time = 65 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 15
                        print(f"   [API Quota Pacing] {err_str[:70]}... Waiting {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        raise e
                else:
                    raise e

        # Synthesize final execution summary from tool traces
        rule_checks = [t for t in tool_call_trace if t['tool'] == 'check_retention_rules']
        last_rule_check = rule_checks[-1]['output'] if rule_checks else {'approved': False, 'reason': 'No rule check performed'}
        
        roi_calls = [t for t in tool_call_trace if t['tool'] == 'calculate_intervention_roi']
        last_roi = roi_calls[-1]['output'] if roi_calls else None

        msg_calls = [t for t in tool_call_trace if t['tool'] == 'generate_retention_message']
        last_msg = msg_calls[-1]['output'] if msg_calls else None

        log_calls = [t for t in tool_call_trace if t['tool'] == 'log_retention_action']
        is_logged = len(log_calls) > 0 and last_rule_check.get('approved', False)

        if not last_rule_check.get('approved', False):
            if churn_prob < 0.35:
                status_str = "HALTED_SAFE_USER"
            else:
                status_str = "HALTED_GUARDRAIL_REJECTED"
        else:
            status_str = "EXECUTED_SUCCESS"

        return {
            'user_id': uid,
            'churn_probability': churn_prob,
            'risk_tier': user_profile.get('risk_tier', 'Unknown'),
            'plan_tier': plan_tier,
            'calculated_ltv': tool_call_trace[0]['output']['calculated_ltv'] if tool_call_trace and tool_call_trace[0]['tool'] == 'calculate_ltv' else monthly_fee * 12,
            'gemini_reasoning_summary': response.text if response else '',
            'guardrail_approved': last_rule_check.get('approved', False),
            'guardrail_reason': last_rule_check.get('reason', ''),
            'roi_simulation': last_roi,
            'generated_message': last_msg,
            'tool_call_trace': tool_call_trace,
            'execution_status': status_str
        }
