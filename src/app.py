import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st

# Ensure project root is in Python path
sys.path.append(os.getcwd())

# Helper function to load .env safely into os.environ
def _load_env_file(env_path: str = ".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

_load_env_file()

def get_gemini_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    _load_env_file()
    return os.environ.get("GEMINI_API_KEY")

from src.agent.agent import RetentionAgent
from src.agent.tools import check_retention_rules, calculate_ltv, calculate_intervention_roi

# Streamlit Page Configuration
st.set_page_config(
    page_title="ChurnGuard AI — Predictive Churn & Retention Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Aesthetics
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.2rem; text-align: center; }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #0F172A; }
    .metric-lbl { font-size: 0.85rem; color: #64748B; font-weight: 600; text-transform: uppercase; }
    .status-badge-high { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .status-badge-med { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .status-badge-low { background-color: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; }
    .card-box { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Data Loading Function: Loads All 3,000 Users from Feature Store as Single Source of Truth
@st.cache_data
def load_dashboard_data():
    feature_store_path = "data/churn_feature_store.csv"
    shap_global_path = "data/shap_global_importance.csv"
    impact_csv_path = "data/retention_impact_simulation.csv"
    impact_json_path = "data/retention_impact_summary.json"
    model_path = "src/models/saved_models/xgboost_model.joblib"

    # 1. Authoritative 3,000-User Feature Store
    df_features = pd.read_csv(feature_store_path)
    
    df_shap_global = pd.read_csv(shap_global_path) if os.path.exists(shap_global_path) else None
    df_sim = pd.read_csv(impact_csv_path) if os.path.exists(impact_csv_path) else None

    summary_json = None
    if os.path.exists(impact_json_path):
        with open(impact_json_path, 'r') as f:
            summary_json = json.load(f)

    # 2. Compute Predictions & SHAP Explanations across all Feature Store Users
    df_combined = df_features.copy()
    
    if os.path.exists(model_path):
        xgb = joblib.load(model_path)
        feature_names = list(xgb.get_booster().feature_names)
        
        # One-hot encode plan_tier for model inference
        df_encoded = pd.get_dummies(df_features, columns=['plan_tier'], dtype=int)
        for col in ['plan_tier_Basic', 'plan_tier_Pro', 'plan_tier_Enterprise']:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        X = df_encoded[feature_names]
        df_combined['churn_probability'] = xgb.predict_proba(X)[:, 1]
        
        # Merge pre-computed local SHAP explanations if available
        if os.path.exists("data/shap_local_explanations.csv"):
            df_shap_local = pd.read_csv("data/shap_local_explanations.csv")
            df_combined = pd.merge(
                df_combined,
                df_shap_local[['user_id', 'driver_1_feature', 'driver_1_shap', 'driver_1_text', 'driver_2_feature', 'driver_2_shap', 'driver_2_text', 'driver_3_feature', 'driver_3_shap', 'driver_3_text']],
                on='user_id',
                how='left'
            )
            
        # Dynamically compute SHAP drivers for any user missing pre-computed text
        explainer = shap.TreeExplainer(xgb)
        shap_vals_matrix = explainer.shap_values(X)
        
        if 'driver_1_feature' not in df_combined.columns:
            for col in ['driver_1_feature', 'driver_1_shap', 'driver_1_text', 'driver_2_feature', 'driver_2_shap', 'driver_2_text', 'driver_3_feature', 'driver_3_shap', 'driver_3_text']:
                df_combined[col] = None

        missing_mask = df_combined['driver_1_feature'].isna()
        if missing_mask.any():
            for idx in df_combined[missing_mask].index:
                s_vals = shap_vals_matrix[idx]
                top_indices = np.argsort(np.abs(s_vals))[::-1][:3]
                for rank_idx, feat_idx in enumerate(top_indices, 1):
                    feat_name = feature_names[feat_idx]
                    s_val = float(s_vals[feat_idx])
                    raw_val = float(X.iloc[idx][feat_name])
                    
                    df_combined.at[idx, f'driver_{rank_idx}_feature'] = feat_name
                    df_combined.at[idx, f'driver_{rank_idx}_shap'] = s_val
                    sign_str = "UP (+)" if s_val > 0 else "DOWN (-)"
                    df_combined.at[idx, f'driver_{rank_idx}_text'] = f"{feat_name} ({raw_val}) pushed churn risk {sign_str} by {s_val:+.2f}"
    else:
        df_combined['churn_probability'] = df_combined['churn_label'].astype(float)

    # 3. Assign Risk Tiers
    def assign_risk_tier(prob):
        if prob >= 0.70: return "High Risk"
        elif prob >= 0.35: return "Medium Risk"
        else: return "Low Risk"

    df_combined['risk_tier'] = df_combined['churn_probability'].apply(assign_risk_tier)
    return df_features, df_combined, df_shap_global, df_sim, summary_json

# Sidebar Setup
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ ChurnGuard AI")
        st.markdown("**Predictive Churn & Automated Retention Engine**")
        st.divider()

        st.markdown("### ⚙️ System Architecture")
        st.code(
            "SQL Feature Store\n"
            "   ↓\n"
            "XGBoost Classifier (AUC 0.916)\n"
            "   ↓\n"
            "SHAP Explainability Engine\n"
            "   ↓\n"
            "Gemini Retention Agent (LLM)\n"
            "   ↓\n"
            "Python Business Guardrails\n"
            "   ↓\n"
            "Financial ROI Simulator",
            language="text"
        )
        st.divider()

        st.markdown("### 🔑 System Status")
        env_key = get_gemini_api_key()
        if env_key:
            st.success("🟢 Gemini LLM Connected (gemini-3.5-flash-lite)")
        else:
            st.warning("🟡 GEMINI_API_KEY Not Detected in Env")

        st.caption("ChurnGuard AI v1.0 | Production-Grade Enterprise Demo")

# MAIN DASHBOARD APPLICATION
def main():
    render_sidebar()
    df_features, df_combined, df_shap_global, df_sim, summary_json = load_dashboard_data()

    st.markdown('<div class="main-title">🛡️ ChurnGuard AI Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">End-to-End Enterprise Churn Prediction, SHAP Explainability & Guardrail-Enforced AI Retention</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "📊 Tab 1: Executive & Financial Overview",
        "🔍 Tab 2: User Risk Explorer",
        "🤖 Tab 3: Retention Agent Sandbox"
    ])

    # =========================================================================
    # TAB 1: EXECUTIVE OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown("### 📈 Portfolio Executive Performance Summary")
        
        # Calculate Key Executive Metrics
        total_analyzed = len(df_combined)
        actual_churners = df_features['churn_label'].sum()
        baseline_churn_rate = (actual_churners / len(df_features)) * 100.0

        if summary_json:
            pm = summary_json['portfolio_metrics']
            targeted_high_risk = pm['eligible_high_risk_users_targeted']
            total_ltv_targeted = pm['total_ltv_of_targeted_users']
            val_saved = pm['total_expected_value_saved']
            cost = pm['total_intervention_cost']
            net_impact = pm['total_net_impact']
            roi_pct = pm['overall_program_roi_percentage']
        else:
            targeted_high_risk = len(df_combined[df_combined['churn_probability'] >= 0.35])
            total_ltv_targeted = df_combined[df_combined['churn_probability'] >= 0.35]['monthly_fee'].sum() * 12
            val_saved = total_ltv_targeted * 0.30
            cost = total_ltv_targeted * 0.15 + (targeted_high_risk * 1.0)
            net_impact = val_saved - cost
            roi_pct = (net_impact / cost * 100) if cost > 0 else 0

        # Top Executive Metrics Row 1
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Users Analyzed", f"{total_analyzed:,}")
        with m2:
            st.metric("Synthetic Baseline Churn Rate", f"{baseline_churn_rate:.2f}%")
        with m3:
            st.metric("Targeted High-Risk Accounts", f"{targeted_high_risk:,}")
        with m4:
            st.metric("Net Financial Impact", f"+${net_impact:,.2f}")

        # Top Executive Metrics Row 2
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.metric("Total LTV Targeted", f"${total_ltv_targeted:,.2f}")
        with f2:
            st.metric("Expected Value Saved (30% Success)", f"${val_saved:,.2f}")
        with f3:
            st.metric("Total Intervention Cost", f"${cost:,.2f}")
        with f4:
            st.metric("Retention Program ROI", f"{roi_pct:.2f}%")

        st.divider()

        # Visualizations Row: Risk Distribution & Global SHAP Drivers
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 🎯 Customer Churn Risk Breakdown")
            risk_counts = df_combined['risk_tier'].value_counts()
            
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            colors = ['#EF4444', '#F59E0B', '#10B981']
            ax_pie.pie(
                risk_counts.values,
                labels=risk_counts.index,
                autopct='%1.1f%%',
                colors=colors[:len(risk_counts)],
                startangle=140,
                explode=[0.05] * len(risk_counts)
            )
            ax_pie.set_title("Risk Tier Distribution", fontsize=11, fontweight='bold')
            st.pyplot(fig_pie)

        with c_right:
            st.markdown("#### 🌐 Global SHAP Feature Importance Ranking")
            if df_shap_global is not None:
                top10_shap = df_shap_global.head(8).sort_values(by='mean_abs_shap', ascending=True)
                fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
                ax_bar.barh(top10_shap['feature'], top10_shap['mean_abs_shap'], color='#3B82F6')
                ax_bar.set_xlabel("Mean |SHAP| Impact", fontsize=10)
                ax_bar.set_title("Top Churn Drivers Across Platform", fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_bar)
            else:
                st.info("Global SHAP data table available in data/shap_global_importance.csv")

        st.divider()

        # Business Interpretation Box
        with st.expander("💡 **Business Interpretation of Key Global Drivers (Click to Expand)**", expanded=True):
            st.markdown("""
            * **Rank 1: `sessions_recent_14d` (Mean |SHAP| = 2.4569):** Recent 14-day login volume is the single strongest indicator of customer health. When recent sessions drop to zero, churn risk spikes dramatically.
            * **Rank 2: `support_ticket_count` (Mean |SHAP| = 0.4229):** Customer support friction serves as the primary early vector of abandonment. Unresolved tickets accelerate churn even among high-paying tiers.
            * **Rank 3: `days_since_last_session` (Mean |SHAP| = 0.3541):** Recency gaps beyond 14 consecutive days strongly correlate with account dormancy and future churn.
            """)

        # Scenario Comparison Table
        st.markdown("#### ⚖️ Portfolio Scenario Financial Comparison")
        st.caption("*Note: Figures represent expected-value simulation projections assuming a 30% intervention conversion rate.*")
        
        sc_data = {
            "Scenario": ["Scenario A: No Intervention (Baseline)", "Scenario B: Target Eligible High-Risk (ChurnGuard AI)"],
            "Targeted Users": [0, targeted_high_risk],
            "Expected Value Saved ($)": ["$0.00", f"${val_saved:,.2f}"],
            "Total Cost ($)": ["$0.00", f"${cost:,.2f}"],
            "Net Impact ($)": ["$0.00", f"+${net_impact:,.2f}"],
            "Program ROI (%)": ["0.00%", f"{roi_pct:.2f}%"]
        }
        st.table(pd.DataFrame(sc_data))

    # =========================================================================
    # TAB 2: USER RISK EXPLORER
    # =========================================================================
    with tab2:
        st.markdown("### 🔍 Customer Account Deep-Dive & Local SHAP Drivers")
        
        user_input_str = st.text_input(
            "Enter User ID:",
            value="3510",
            placeholder="e.g. 3510 or 2410",
            key="tab2_user_id_input"
        )

        # Validate User ID Input against Feature Store Single Source of Truth
        try:
            selected_user_id = int(user_input_str.strip()) if user_input_str.strip() else None
        except ValueError:
            st.error("⚠️ Invalid User ID format. Please enter a numeric User ID.")
            selected_user_id = None

        if selected_user_id is not None:
            user_matches = df_combined[df_combined['user_id'] == selected_user_id]
            if user_matches.empty:
                st.warning("⚠️ User ID not found in feature store.")
            else:
                user_row = user_matches.iloc[0]
                
                # Risk Badge Styling
                r_tier = user_row['risk_tier']
                badge_class = "status-badge-high" if r_tier == "High Risk" else ("status-badge-med" if r_tier == "Medium Risk" else "status-badge-low")

                st.markdown(f"#### Account Profile: User #{selected_user_id} <span class='{badge_class}'>{r_tier}</span>", unsafe_allow_html=True)
                st.write("")

                # User Key Info Cards
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1: st.metric("Plan Tier", user_row['plan_tier'])
                with k2: st.metric("Monthly Fee", f"${user_row['monthly_fee']:.2f}")
                with k3: st.metric("Predicted Churn Risk", f"{user_row['churn_probability']*100:.1f}%")
                with k4: st.metric("Calculated LTV", f"${user_row['monthly_fee']*12:,.2f}")
                with k5: st.metric("Actual Target Label", "Churned (1)" if user_row.get('churn_label', user_row.get('actual_churn', 0)) == 1 else "Active (0)")

                st.divider()

                # Behavioral Features Table
                st.markdown("#### 📋 Behavioral Feature Snapshots (Days 1–60 Cutoff Window)")
                b1, b2, b3, b4, b5, b6 = st.columns(6)
                with b1: st.metric("Recent 14d Logins", f"{user_row['sessions_recent_14d']:.0f}")
                with b2: st.metric("Recency Gap (Days)", f"{user_row['days_since_last_session']:.0f}")
                with b3: st.metric("Session Drop %", f"{user_row['session_drop_pct']:.1f}%")
                with b4: st.metric("Support Tickets", f"{user_row['support_ticket_count']:.0f}")
                with b5: st.metric("Unresolved Tickets", f"{user_row['unresolved_tickets']:.0f}")
                with b6: st.metric("Core Feature Usage", f"{user_row['core_feature_usage_count']:.0f}")

                st.divider()

                # Local SHAP Explanation Table & Plot
                st.markdown("#### 🧬 Top 3 Local SHAP Risk Drivers for User #" + str(selected_user_id))
                st.caption("Positive SHAP (+) increases churn risk | Negative SHAP (-) pulls score toward retention")

                shap_drivers_data = []
                for i in range(1, 4):
                    f_col = f"driver_{i}_feature"
                    s_col = f"driver_{i}_shap"
                    t_col = f"driver_{i}_text"
                    if f_col in user_row and pd.notna(user_row[f_col]):
                        feat_name = str(user_row[f_col])
                        shap_val = float(user_row[s_col])
                        text_exp = str(user_row[t_col])
                        direction = "Pushes Churn Risk UP (+)" if shap_val > 0 else "Pulls Churn Risk DOWN (-)"
                        shap_drivers_data.append({
                            "Driver Rank": f"Driver {i}",
                            "Feature": feat_name,
                            "SHAP Impact": f"{shap_val:+.4f}",
                            "Direction": direction,
                            "Explanation": text_exp
                        })

                st.table(pd.DataFrame(shap_drivers_data))

                # Local SHAP Plot Image or Dynamic Plot
                waterfall_img_path = f"src/explainability/shap_user_{selected_user_id}_waterfall.png"
                if os.path.exists(waterfall_img_path):
                    st.image(waterfall_img_path, caption=f"Local SHAP Waterfall Explanation for User #{selected_user_id}", width=800)
                else:
                    fig_loc, ax_loc = plt.subplots(figsize=(8, 3))
                    feats = [d['Feature'] for d in shap_drivers_data]
                    s_vals = [float(d['SHAP Impact']) for d in shap_drivers_data]
                    colors_loc = ['#EF4444' if v > 0 else '#10B981' for v in s_vals]
                    ax_loc.barh(feats, s_vals, color=colors_loc)
                    ax_loc.axvline(0, color='gray', linestyle='--')
                    ax_loc.set_title(f"User #{selected_user_id} - Top 3 SHAP Drivers", fontsize=10, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig_loc)

    # =========================================================================
    # TAB 3: RETENTION AGENT SANDBOX
    # =========================================================================
    with tab3:
        st.markdown("### 🤖 Guardrail-Enforced AI Retention Agent Sandbox")
        st.caption("Executes real Gemini LLM reasoning + deterministic Python business tools (LTV, Rule Caps, ROI, Messaging, Logging).")

        sandbox_user_input_str = st.text_input(
            "Enter User ID for Agent Execution:",
            value="3510",
            placeholder="e.g. 3510 or 2410",
            key="tab3_user_id_input"
        )

        # Validate User ID Input against Feature Store Single Source of Truth
        try:
            sandbox_user_id = int(sandbox_user_input_str.strip()) if sandbox_user_input_str.strip() else None
        except ValueError:
            st.error("⚠️ Invalid User ID format. Please enter a numeric User ID.")
            sandbox_user_id = None

        if sandbox_user_id is not None:
            sb_matches = df_combined[df_combined['user_id'] == sandbox_user_id]
            if sb_matches.empty:
                st.warning("⚠️ User ID not found in feature store.")
            else:
                sb_row = sb_matches.iloc[0]
                sb_churn_prob = float(sb_row['churn_probability'])
                sb_plan = str(sb_row['plan_tier'])
                sb_fee = float(sb_row['monthly_fee'])

                # Prepare Top SHAP drivers structure
                sb_top_drivers = []
                for i in range(1, 4):
                    if f"driver_{i}_feature" in sb_row and pd.notna(sb_row[f"driver_{i}_feature"]):
                        sb_top_drivers.append({
                            'feature': str(sb_row[f"driver_{i}_feature"]),
                            'shap_value': float(sb_row[f"driver_{i}_shap"]),
                            'text': str(sb_row[f"driver_{i}_text"])
                        })

                # Display Structured Risk Profile JSON Payload
                st.markdown("#### 📦 Structured User Risk Profile (Payload sent to Gemini)")
                profile_payload = {
                    "user_id": int(sandbox_user_id),
                    "churn_probability": round(sb_churn_prob, 4),
                    "risk_tier": sb_row['risk_tier'],
                    "plan_tier": sb_plan,
                    "monthly_fee": sb_fee,
                    "top_3_shap_drivers": sb_top_drivers
                }
                st.json(profile_payload)

                st.divider()

                # Run Retention Agent Button
                if st.button("🚀 Run Retention Agent for User #" + str(sandbox_user_id), type="primary"):
                    env_key = get_gemini_api_key()
                    if not env_key:
                        st.error("⚠️ GEMINI_API_KEY environment variable is missing. Please configure your API key in environment or .env file.")
                    else:
                        with st.spinner(f"Agent analyzing customer profile #{sandbox_user_id} & calling Python tools..."):
                            try:
                                agent = RetentionAgent(model_name="gemini-3.5-flash-lite")
                                result = agent.process_user_risk_profile(profile_payload)

                                st.success("✅ Retention Agent Execution Completed Successfully!")

                                # 1. Gemini Reasoning Output Box
                                st.markdown("#### 🧠 Gemini LLM Strategic Reasoning Summary")
                                st.info(result['gemini_reasoning_summary'])

                                # 2. Tool Call Sequence Trace
                                with st.expander("🛠️ **View Deterministic Python Tool Call Sequence Trace**", expanded=True):
                                    for idx, call in enumerate(result['tool_call_trace'], 1):
                                        st.write(f"**Step {idx}: Call [`{call['tool']}`]**")
                                        st.json({"input": call['input'], "output": call['output']})

                                # 3. Guardrail Authorization & Action Results
                                st.markdown("#### 🔒 Policy Guardrail & Financial ROI Results")
                                g_col1, g_col2 = st.columns(2)

                                with g_col1:
                                    st.write(f"**Guardrail Approved:** `{result['guardrail_approved']}`")
                                    st.write(f"**Policy Reason:** {result['guardrail_reason']}")
                                    st.write(f"**Calculated LTV:** `${result['calculated_ltv']:,.2f}`")
                                    st.write(f"**Execution Status:** `{result['execution_status']}`")

                                with g_col2:
                                    if result['roi_simulation']:
                                        roi = result['roi_simulation']
                                        st.write(f"**Expected Value Saved:** `${roi['expected_value_saved']:,.2f}`")
                                        st.write(f"**Intervention Cost:** `${roi['intervention_cost']:,.2f}`")
                                        st.write(f"**Net Financial Impact:** `${roi['net_impact']:,.2f}`")
                                        st.write(f"**Net ROI Percentage:** `{roi['roi_percentage']:.2f}%`")

                                # 4. Generated Customer Retention Message Copy
                                if result['generated_message']:
                                    msg = result['generated_message']
                                    st.markdown("#### ✉️ Generated Customer Communication Copy")
                                    st.markdown(f"**Subject:** `{msg['subject_line']}`")
                                    st.markdown(f"**Body:** *\"{msg['message_body']}\"*")

                            except Exception as e:
                                st.error(f"❌ Error during Retention Agent execution: {str(e)}")

if __name__ == "__main__":
    main()
