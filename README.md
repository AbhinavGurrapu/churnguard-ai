# ChurnGuard AI — Predictive Churn & Automated Retention Engine

## Objective
ChurnGuard AI is an end-to-end predictive customer retention engine designed to bridge raw product analytics, machine learning, and automated business intervention. The primary goal of the system is to identify behavioral product churn early and trigger guardrail-constrained retention actions that protect customer lifetime value (LTV).

> [!NOTE]
> **Project Status (Complete - Days 1–6):** The full 6-day production-inspired pipeline is fully implemented, validated, and interactive via a 3-tab Streamlit Dashboard (`src/app.py`).

---

## 🏛️ End-to-End System Architecture

```text
PostgreSQL Feature Store  (Temporal Cutoff Days 1–60)
        ↓
XGBoost Churn Prediction  (ROC-AUC 0.9162 on 600 Test Users)
        ↓
SHAP Explainability       (TreeExplainer Margin Space & Local Risk Profiles)
        ↓
Gemini Retention Agent    (Real LLM Tool Calling via google-genai)
        ↓
Python Business Rules     (Hard Policy Caps: Basic ≤10%, Pro ≤15%, Enterprise ≤20%)
        ↓
ROI & Impact Simulator   (Expected Value Saved & Net Financial Impact $)
        ↓
Streamlit Dashboard       (Executive Overview, Risk Explorer, Agent Sandbox)
```

---

## 🚀 How to Run the Application

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ and PostgreSQL are installed. Create a `.env` file in the project root to configure your Gemini API Key:

```bash
# .env (Ignored by Git)
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 2. Run the Streamlit Dashboard
Launch the interactive 3-tab Streamlit dashboard:

```bash
streamlit run src/app.py
```

### 3. Execute Verification Audit Suites
To run the automated validation test suites across all pipeline stages:

```bash
# Day 1: Feature Store & Fan-out Validation Audit
python src/data/validate_feature_store.py

# Day 2: ML Pipeline & Data Leakage Audit
python src/models/verify_day2_eval.py

# Day 3: SHAP Margin Space & Sign Audit
python src/explainability/verify_day3_global.py

# Day 4: Agent Architecture & Security Audit
python src/agent/verify_day4_agent.py

# Day 5: Portfolio Business Impact Simulator Test
python src/business/test_impact_simulator.py
```

---

## 📊 Dataset & Temporal Feature Architecture

### Synthetic Dataset & Rationale
Many commonly used churn datasets are already aggregated at the customer level and do not provide the event-level temporal history needed for this project.
* A realistic synthetic event stream was generated spanning **90 days** across **3,000 users**.
* User activity incorporates 3 latent probabilistic cohorts (*Engaged*, *Gradual Decline*, and *Friction/Support Drop-off*).

### Relational Schema (PostgreSQL)
1. `users` (3,000 rows): `user_id`, `signup_date`, `plan_tier`, `monthly_fee` ($29, $99, $299)
2. `sessions` (147,733 rows): `session_id`, `user_id`, `session_start`, `session_end`, `device_type`
3. `events` (664,819 rows): `event_id`, `session_id`, `user_id`, `event_timestamp`, `event_name`
4. `orders` (1,667 rows): `order_id`, `user_id`, `order_timestamp`, `amount`
5. `support_tickets` (1,015 rows): `ticket_id`, `user_id`, `created_at`, `category`, `status`

### Temporal Isolation & Leakage Prevention
```text
       Observation Window (Days 1 to 60)         Future Label Window (Days 61 to 90)
 ┌───────────────────────────────────────────────┬───────────────────────────────┐
 │  Jan 1, 2026 00:00:00 to March 1, 2026 23:59 │ March 2, 2026 to March 31, 2026│
 │                                               │                               │
 │  SQL extracts ALL features strictly from      │  Target label evaluated HERE: │
 │  events occurring within this 60-day window. │  Has user logged any sessions │
 │                                               │  or orders in this 30-day     │
 │                                               │  future window?               │
 │                                               │   NO  ==> churn_label = 1     │
 │                                               │   YES ==> churn_label = 0     │
 └───────────────────────────────────────────────┴───────────────────────────────┘
                                                 ▲
                                          PREDICTION POINT
                                      (March 1, 2026 23:59:59)
```

---

## 🤖 Guardrail-Enforced AI Agent Architecture

The `RetentionAgent` (`src/agent/agent.py`) combines Google Gemini (`gemini-3.5-flash-lite` via `google-genai`) with 5 deterministic Python tools:

1. `calculate_ltv()`: Computes annual customer LTV ($\text{monthly\_fee} \times 12$).
2. `check_retention_rules()`: Enforces hard business policy caps (Basic $\le 10\%$, Pro $\le 15\%$, Enterprise $\le 20\%$) and risk threshold ($P \ge 35\%$).
3. `calculate_intervention_roi()`: Calculates financial expected value saved, intervention costs, and net ROI $\%$.
4. `generate_retention_message()`: Constructs targeted copy based on SHAP risk vector.
5. `log_retention_action()`: Records execution status in `data/retention_action_logs.csv`.

---

## 📈 Portfolio Business Impact & ROI Simulation

Across the 600 held-out test users:
* **Targeted High-Risk Accounts ($P \ge 35\%$):** 84 accounts (14.0%)
* **Total LTV Targeted:** $86,832.00
* **Expected Value Saved (30% retention rate assumption):** $26,049.60
* **Total Intervention Cost:** $14,137.20
* **Net Financial Impact:** **+$11,912.40**
* **Retention Program Overall ROI:** **84.26%**

---

## 🎯 Key Interview Talking Points

1. **How Join Fan-Out Was Prevented:** Used 4 independent CTEs to aggregate sub-tables down to 1 row per user *before* joining back to `users`, guaranteeing $1 \text{ row} = 1 \text{ user}$.
2. **How Data Leakage Was Prevented:** Strict SQL cutoff at `timestamp <= '2026-03-01 23:59:59'`. Preprocessing scalers and imputers fit strictly on `X_train` (2,400 users) and applied to `X_test` (600 users).
3. **Why Business Rules are Deterministic:** LLMs are non-deterministic text generators. By placing policy caps in Python tools, the LLM can recommend actions, but Python tools act as an unbypassable policy firewall.
4. **SHAP TreeExplainer Margin Space:** SHAP values explain XGBoost outputs in raw log-odds margin space. Base value $-2.0542$ corresponds to $11.36\%$ baseline dataset churn probability.

---

## Project Roadmap

* [x] **Day 1:** Data Architecture, Synthetic Event Stream, PostgreSQL Schema & Feature Store.
* [x] **Day 2:** ML Classification Pipeline (Logistic Regression Baseline vs. XGBoost Classifier & Metrics).
* [x] **Day 3:** SHAP Explainability Engine (Global Feature Importance + Per-User Local Drivers).
* [x] **Day 4:** Guardrail-Enforced AI Retention Agent (LLM Function Calling & Python Logic Tools).
* [x] **Day 5:** ROI / LTV Financial Simulator & Execution Action Logging.
* [x] **Day 6:** 3-Tab Streamlit Dashboard & Interview Defense Setup.
