# ChurnGuard AI — Predictive Churn & Automated Retention Engine

## Objective
ChurnGuard AI is an end-to-end predictive customer retention engine designed to bridge raw product analytics, machine learning, and automated business intervention. The primary goal of the system is to identify behavioral product churn early and trigger guardrail-constrained retention actions that protect customer lifetime value (LTV).

> [!NOTE]
> **Project Status (Day 1 Complete):** The SQL Data Architecture, Synthetic Event Stream, PostgreSQL Schema, and Temporal Feature Store are fully implemented and validated. Machine Learning (XGBoost/Logistic Regression), SHAP explainability, Agentic AI retention tools, and the Streamlit dashboard are planned for upcoming implementation phases.

---

## Synthetic Dataset & Rationale
Many commonly used churn datasets are already aggregated at the customer level and do not provide the event-level temporal history needed for this project.

To showcase production-inspired temporal SQL feature store construction:
* A realistic synthetic event stream was generated spanning **90 days** across **3,000 users**.
* All users signed up in December 2025 prior to the observation window start date, ensuring a complete 60-day feature history for every record.
* User activity incorporates 3 latent probabilistic cohorts (*Engaged*, *Gradual Decline*, and *Friction/Support Drop-off*), allowing churn to emerge naturally rather than via hardcoded rules.

---

## Dataset Schema & Tables

The raw relational database comprises 5 core tables hosted in **PostgreSQL**:

1. `users` (3,000 rows): `user_id`, `signup_date`, `plan_tier`, `monthly_fee` (29.00, 99.00, 299.00)
2. `sessions` (147,733 rows): `session_id`, `user_id`, `session_start`, `session_end`, `device_type`
3. `events` (664,819 rows): `event_id`, `session_id`, `user_id`, `event_timestamp`, `event_name`
4. `orders` (1,667 rows): `order_id`, `user_id`, `order_timestamp`, `amount`
5. `support_tickets` (1,015 rows): `ticket_id`, `user_id`, `created_at`, `category`, `status`

---

## Temporal Cutoff & Behavioral Churn Definition

To guarantee **temporal isolation** and prevent future-window data leakage, the timeline is strictly divided into two windows:

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

* **Behavioral Churn Definition:** A user is labeled `churn_label = 1` (Behavioral Churn / Product Inactivity) if they log **0 sessions AND 0 orders** in Days 61–90. Otherwise, `churn_label = 0`.

---

## PostgreSQL Feature Store Architecture & Design Decisions

### 1. Preventing SQL Join Fan-Out
Naively joining `sessions`, `events`, `orders`, and `support_tickets` directly before running aggregations creates a multi-table row multiplication problem. 

To solve this, `sql/02_feature_store.sql` builds **4 independent user-level Common Table Expressions (CTEs)** (`session_features`, `event_features`, `order_features`, `support_features`) that aggregate metrics down to **1 row per user** *before* joining back to `users`. This guarantees that $1 \text{ row} = 1 \text{ user}$ without row inflation.

### 2. Preventing Data Leakage
All feature extraction queries filter strictly on `timestamp <= '2026-03-01 23:59:59'`. The model inputs never observe any event occurring in the Days 61–90 label window.

### 3. PostgreSQL Window Function Integration
The feature `avg_session_gap_days` utilizes `LAG(session_start) OVER (PARTITION BY user_id ORDER BY session_start)` inside a CTE to compute the mean day gap between consecutive sessions for each user.

---

## Current SQL Feature Dictionary

| Feature Name | Category | Logic & Business Rationale | Missing / Sentinel Rule |
| :--- | :--- | :--- | :--- |
| `days_since_last_session` | Recency | Days from user's last session to Day 60 cutoff | Sentinel `60` if 0 sessions |
| `sessions_recent_14d` | Engagement | Session count in Days 47–60 | `0` if no sessions |
| `sessions_previous_14d` | Engagement | Session count in Days 33–46 | `0` if no sessions |
| `session_drop_pct` | Trend | $\frac{\text{prev\_14d} - \text{rec\_14d}}{\text{prev\_14d}} \times 100$ | `0.0%` if both 0; `100.0%` if dropped to 0 |
| `total_events_60d` | Engagement | Total event count over 60 days | `0` if no events |
| `core_feature_usage_count`| Adoption | Count of `'feature_used'` events | `0` if none used |
| `avg_session_duration_minutes`| Depth | Mean duration in minutes | `0.0` if no sessions |
| `avg_session_gap_days` | Velocity | Mean gap via `LAG(session_start)` | **`NULL`** if $< 2$ sessions (handled in ML) |
| `monthly_fee` | Value | Subscription tier price (29.00, 99.00, 299.00) | Raw fee |
| `total_spend_60d` | Value | `(monthly_fee * 2) + sum(add_on_orders)` | Base fee sum |
| `support_ticket_count` | Friction | Total support tickets in Days 1–60 | `0` if no tickets |
| `unresolved_tickets` | Friction | Count of 'open' or 'escalated' tickets | `0` if no unresolved tickets |
| **`churn_label`** | **Target** | **1 if inactive in Days 61–90, else 0** | **Binary Target** |

---

## Validation & Audit Results

The feature store script `src/data/validate_feature_store.py` executed a 4-point audit:

* **Row Count Audit:** Exactly **3,000 unique rows** for 3,000 users ($0$ fan-out).
* **Target Distribution:** **2,634 Active (87.80%)**, **366 Churned (12.20%)** (Synthetic churn rate of 12.20%).
* **Manual Feature Spot-Checks:** Manual spot-checks matched the independently calculated feature values for sample users (`1001`, `1002`, `1003`).
* **Behavioral Separation Verification:**
  * Non-churned users average **3.15 days** since last session vs. **14.92 days** for churned users.
  * Churned users exhibit an average **65.38% session volume drop** in the final 14 days vs. **5.53%** for active users.

---

## Project Directory Structure

```text
churnguard-ai/
├── README.md
├── .gitignore
├── data/
│   ├── users.csv                       (3,000 raw users)
│   ├── sessions.csv                    (147,733 raw sessions)
│   ├── events.csv                      (664,819 raw events)
│   ├── orders.csv                      (1,667 raw orders)
│   ├── support_tickets.csv             (1,015 raw support tickets)
│   └── churn_feature_store.csv         (Validated 1-row-per-user ML Feature Store)
├── sql/
│   ├── 01_schema.sql                   (PostgreSQL DDL script with indexes)
│   └── 02_feature_store.sql            (1-row-per-user SQL query using CTEs & window functions)
└── src/
    └── data/
        ├── generate_data.py            (Synthetic event generator)
        ├── load_to_postgres.py         (PostgreSQL data loader)
        └── validate_feature_store.py   (4-point validation audit script)
```

---

## Technologies Used (Day 1)

* **Database:** PostgreSQL (SQL DDL, CTEs, Window Functions, Indexes)
* **Language & Libraries:** Python 3.13, Pandas, NumPy, psycopg2-binary
* **Environment:** Local PostgreSQL + pgAdmin

---

## Project Roadmap

* [x] **Day 1:** Data Architecture, Synthetic Event Stream, PostgreSQL Schema & Feature Store.
* [x] **Day 2:** ML Classification Pipeline (Logistic Regression Baseline vs. XGBoost Classifier & Metrics).
* [x] **Day 3:** SHAP Explainability Engine (Global Feature Importance + Per-User Local Drivers).
* [x] **Day 4:** Guardrail-Enforced AI Retention Agent (LLM Function Calling & Python Logic Tools).
* [ ] **Day 5:** ROI / LTV Financial Simulator & Execution Action Logging.
* [ ] **Day 6:** 3-Tab Streamlit Dashboard & Interview Defense Setup.
