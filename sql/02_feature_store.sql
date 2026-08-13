-- 02_feature_store.sql: PostgreSQL Feature Store Query for ChurnGuard AI
-- Prevents SQL Join Fan-out using 1-row-per-user CTEs before joining to users.
-- Strictly isolates Days 1-60 for feature extraction and Days 61-90 for target labeling.

DROP TABLE IF EXISTS churn_feature_store CASCADE;

CREATE TABLE churn_feature_store AS
WITH 

-- CTE 1: Session Features (Days 1-60 strictly)
-- Computes Recency, Recent vs Previous 14d Frequency, Session Drop %, Avg Duration & Avg Gap (Window Function)
raw_session_gaps AS (
    SELECT 
        user_id,
        session_start,
        LAG(session_start) OVER (PARTITION BY user_id ORDER BY session_start) AS prev_session_start
    FROM sessions
    WHERE session_start <= '2026-03-01 23:59:59'
),
user_session_gaps AS (
    SELECT 
        user_id,
        AVG(EXTRACT(EPOCH FROM (session_start - prev_session_start)) / 86400.0) AS raw_avg_gap
    FROM raw_session_gaps
    WHERE prev_session_start IS NOT NULL
    GROUP BY user_id
),
session_features AS (
    SELECT 
        s.user_id,
        -- Days since last session (Sentinel 60 if 0 sessions)
        COALESCE(EXTRACT(DAY FROM ('2026-03-01 23:59:59'::timestamp - MAX(s.session_start))), 60)::INT AS days_since_last_session,
        
        -- Sessions in recent 14 days (Days 47-60: Feb 16 to Mar 1)
        COUNT(s.session_id) FILTER (WHERE s.session_start >= '2026-02-16 00:00:00')::INT AS sessions_recent_14d,
        
        -- Sessions in previous 14 days (Days 33-46: Feb 2 to Feb 15)
        COUNT(s.session_id) FILTER (WHERE s.session_start BETWEEN '2026-02-02 00:00:00' AND '2026-02-15 23:59:59')::INT AS sessions_previous_14d,
        
        -- Average session duration in minutes
        COALESCE(ROUND(AVG(EXTRACT(EPOCH FROM (s.session_end - s.session_start)) / 60.0)::numeric, 2), 0.0) AS avg_session_duration_minutes,
        
        -- Window function derived gap (Sentinel 60.0 if <= 1 session)
        ROUND(sg.raw_avg_gap::numeric, 2) AS avg_session_gap_days
    FROM sessions s
    LEFT JOIN user_session_gaps sg ON s.user_id = sg.user_id
    WHERE s.session_start <= '2026-03-01 23:59:59'
    GROUP BY s.user_id, sg.raw_avg_gap
),

-- CTE 2: Event Features (Days 1-60 strictly)
event_features AS (
    SELECT 
        user_id,
        COUNT(event_id)::INT AS total_events_60d,
        COUNT(event_id) FILTER (WHERE event_name = 'feature_used')::INT AS core_feature_usage_count
    FROM events
    WHERE event_timestamp <= '2026-03-01 23:59:59'
    GROUP BY user_id
),

-- CTE 3: Order Features (Days 1-60 strictly)
order_features AS (
    SELECT 
        user_id,
        COALESCE(SUM(amount), 0.0)::NUMERIC(8,2) AS add_on_spend_60d
    FROM orders
    WHERE order_timestamp <= '2026-03-01 23:59:59'
    GROUP BY user_id
),

-- CTE 4: Support Ticket Features (Days 1-60 strictly)
support_features AS (
    SELECT 
        user_id,
        COUNT(ticket_id)::INT AS support_ticket_count,
        COUNT(ticket_id) FILTER (WHERE status IN ('open', 'escalated'))::INT AS unresolved_tickets
    FROM support_tickets
    WHERE created_at <= '2026-03-01 23:59:59'
    GROUP BY user_id
),

-- CTE 5: Future Window Target Label (Days 61-90 strictly)
-- Churn label = 1 if user logged 0 sessions AND 0 orders in Days 61-90
future_activity AS (
    SELECT user_id FROM sessions WHERE session_start >= '2026-03-02 00:00:00'
    UNION
    SELECT user_id FROM orders WHERE order_timestamp >= '2026-03-02 00:00:00'
)

-- FINAL ASSEMBLY: Join 1-row-per-user CTEs back to USERS
SELECT 
    u.user_id,
    u.plan_tier,
    u.monthly_fee,
    
    -- Total Spend = (Monthly Fee * 2) + Add-on Orders Spend
    ROUND((u.monthly_fee * 2 + COALESCE(o.add_on_spend_60d, 0.0))::numeric, 2) AS total_spend_60d,
    
    COALESCE(sf.days_since_last_session, 60) AS days_since_last_session,
    COALESCE(sf.sessions_recent_14d, 0) AS sessions_recent_14d,
    COALESCE(sf.sessions_previous_14d, 0) AS sessions_previous_14d,
    
    -- Session Drop Percentage calculation with NULLIF division by zero protection
    ROUND(
        CASE 
            WHEN COALESCE(sf.sessions_previous_14d, 0) = 0 AND COALESCE(sf.sessions_recent_14d, 0) = 0 THEN 0.0
            WHEN COALESCE(sf.sessions_previous_14d, 0) = 0 THEN -100.0
            ELSE ((sf.sessions_previous_14d - sf.sessions_recent_14d)::numeric / sf.sessions_previous_14d) * 100.0
        END, 2
    ) AS session_drop_pct,
    
    COALESCE(ef.total_events_60d, 0) AS total_events_60d,
    COALESCE(ef.core_feature_usage_count, 0) AS core_feature_usage_count,
    COALESCE(sf.avg_session_duration_minutes, 0.0) AS avg_session_duration_minutes,
    sf.avg_session_gap_days AS avg_session_gap_days,
    
    COALESCE(tf.support_ticket_count, 0) AS support_ticket_count,
    COALESCE(tf.unresolved_tickets, 0) AS unresolved_tickets,
    
    -- Target Label: 1 if user absent from future_activity, else 0
    CASE WHEN fa.user_id IS NULL THEN 1 ELSE 0 END AS churn_label

FROM users u
LEFT JOIN session_features sf ON u.user_id = sf.user_id
LEFT JOIN event_features ef ON u.user_id = ef.user_id
LEFT JOIN order_features o ON u.user_id = o.user_id
LEFT JOIN support_features tf ON u.user_id = tf.user_id
LEFT JOIN future_activity fa ON u.user_id = fa.user_id;

-- CREATE INDEX ON FEATURE STORE
CREATE UNIQUE INDEX idx_feature_store_user ON churn_feature_store(user_id);
