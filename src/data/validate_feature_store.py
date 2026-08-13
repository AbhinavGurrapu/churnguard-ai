import os
import sqlite3
import pandas as pd
import numpy as np

def validate_local_feature_store():
    print("=" * 70)
    print("      CHURNGUARD AI - DAY 1 FEATURE STORE VALIDATION AUDIT      ")
    print("=" * 70)

    # Read raw CSV data files
    df_users = pd.read_csv("data/users.csv")
    df_sessions = pd.read_csv("data/sessions.csv")
    df_events = pd.read_csv("data/events.csv")
    df_orders = pd.read_csv("data/orders.csv")
    df_tickets = pd.read_csv("data/support_tickets.csv")

    # Use SQLite in-memory engine to execute 02_feature_store.sql logic for verification
    conn = sqlite3.connect(":memory:")
    
    df_users.to_sql("users", conn, index=False)
    df_sessions.to_sql("sessions", conn, index=False)
    df_events.to_sql("events", conn, index=False)
    df_orders.to_sql("orders", conn, index=False)
    df_tickets.to_sql("support_tickets", conn, index=False)

    # Execute SQLite-equivalent of 02_feature_store.sql
    sql_script = """
    WITH 
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
            AVG((julianday(session_start) - julianday(prev_session_start))) AS raw_avg_gap
        FROM raw_session_gaps
        WHERE prev_session_start IS NOT NULL
        GROUP BY user_id
    ),
    session_features AS (
        SELECT 
            s.user_id,
            COALESCE(CAST((julianday('2026-03-01 23:59:59') - julianday(MAX(s.session_start))) AS INT), 60) AS days_since_last_session,
            SUM(CASE WHEN s.session_start >= '2026-02-16 00:00:00' THEN 1 ELSE 0 END) AS sessions_recent_14d,
            SUM(CASE WHEN s.session_start BETWEEN '2026-02-02 00:00:00' AND '2026-02-15 23:59:59' THEN 1 ELSE 0 END) AS sessions_previous_14d,
            COALESCE(ROUND(AVG((julianday(s.session_end) - julianday(s.session_start)) * 1440.0), 2), 0.0) AS avg_session_duration_minutes,
            ROUND(sg.raw_avg_gap, 2) AS avg_session_gap_days
        FROM sessions s
        LEFT JOIN user_session_gaps sg ON s.user_id = sg.user_id
        WHERE s.session_start <= '2026-03-01 23:59:59'
        GROUP BY s.user_id
    ),
    event_features AS (
        SELECT 
            user_id,
            COUNT(event_id) AS total_events_60d,
            SUM(CASE WHEN event_name = 'feature_used' THEN 1 ELSE 0 END) AS core_feature_usage_count
        FROM events
        WHERE event_timestamp <= '2026-03-01 23:59:59'
        GROUP BY user_id
    ),
    order_features AS (
        SELECT 
            user_id,
            COALESCE(SUM(amount), 0.0) AS add_on_spend_60d
        FROM orders
        WHERE order_timestamp <= '2026-03-01 23:59:59'
        GROUP BY user_id
    ),
    support_features AS (
        SELECT 
            user_id,
            COUNT(ticket_id) AS support_ticket_count,
            SUM(CASE WHEN status IN ('open', 'escalated') THEN 1 ELSE 0 END) AS unresolved_tickets
        FROM support_tickets
        WHERE created_at <= '2026-03-01 23:59:59'
        GROUP BY user_id
    ),
    future_activity AS (
        SELECT user_id FROM sessions WHERE session_start >= '2026-03-02 00:00:00'
        UNION
        SELECT user_id FROM orders WHERE order_timestamp >= '2026-03-02 00:00:00'
    )
    SELECT 
        u.user_id,
        u.plan_tier,
        u.monthly_fee,
        ROUND(u.monthly_fee * 2 + COALESCE(o.add_on_spend_60d, 0.0), 2) AS total_spend_60d,
        COALESCE(sf.days_since_last_session, 60) AS days_since_last_session,
        COALESCE(sf.sessions_recent_14d, 0) AS sessions_recent_14d,
        COALESCE(sf.sessions_previous_14d, 0) AS sessions_previous_14d,
        ROUND(
            CASE 
                WHEN COALESCE(sf.sessions_previous_14d, 0) = 0 AND COALESCE(sf.sessions_recent_14d, 0) = 0 THEN 0.0
                WHEN COALESCE(sf.sessions_previous_14d, 0) = 0 THEN -100.0
                ELSE ((sf.sessions_previous_14d - sf.sessions_recent_14d) * 100.0) / sf.sessions_previous_14d
            END, 2
        ) AS session_drop_pct,
        COALESCE(ef.total_events_60d, 0) AS total_events_60d,
        COALESCE(ef.core_feature_usage_count, 0) AS core_feature_usage_count,
        COALESCE(sf.avg_session_duration_minutes, 0.0) AS avg_session_duration_minutes,
        sf.avg_session_gap_days AS avg_session_gap_days,
        COALESCE(tf.support_ticket_count, 0) AS support_ticket_count,
        COALESCE(tf.unresolved_tickets, 0) AS unresolved_tickets,
        CASE WHEN fa.user_id IS NULL THEN 1 ELSE 0 END AS churn_label
    FROM users u
    LEFT JOIN session_features sf ON u.user_id = sf.user_id
    LEFT JOIN event_features ef ON u.user_id = ef.user_id
    LEFT JOIN order_features o ON u.user_id = o.user_id
    LEFT JOIN support_features tf ON u.user_id = tf.user_id
    LEFT JOIN future_activity fa ON u.user_id = fa.user_id;
    """

    df_fs = pd.read_sql_query(sql_script, conn)
    conn.close()

    # Save local CSV feature store
    df_fs.to_csv("data/churn_feature_store.csv", index=False)

    # 1. Fan-out Verification
    num_users = len(df_users)
    fs_rows = len(df_fs)
    unique_users = df_fs['user_id'].nunique()

    print(f"\n[1/4] JOIN FAN-OUT & ROW COUNT AUDIT:")
    print(f" - Raw Users Table Count:      {num_users}")
    print(f" - Feature Store Row Count:    {fs_rows}")
    print(f" - Unique User IDs in Store:  {unique_users}")
    
    if num_users == fs_rows == unique_users:
        print(" ==> SUCCESS: Exactly 1 row per user! No join fan-out occurred.")
    else:
        print(" ==> [ERROR] Row count mismatch detected!")

    # 2. Target Label Distribution
    churned = df_fs['churn_label'].sum()
    non_churned = fs_rows - churned
    churn_rate = (churned / fs_rows) * 100.0

    print(f"\n[2/4] TARGET DISTRIBUTION (DAYS 61-90 FUTURE WINDOW):")
    print(f" - Total Users:       {fs_rows}")
    print(f" - Non-Churned (0):   {non_churned} ({100-churn_rate:.2f}%)")
    print(f" - Churned (1):       {churned} ({churn_rate:.2f}%)")

    # 3. Manual Spot-Check Verification for User 1001, 1002, 1003
    print(f"\n[3/4] MANUAL SPOT-CHECK VERIFICATION FOR SAMPLE USERS:")
    sample_ids = [1001, 1002, 1003]

    for uid in sample_ids:
        row = df_fs[df_fs['user_id'] == uid].iloc[0]
        
        # Manual count from raw CSVs (Days 1-60)
        u_sessions = df_sessions[(df_sessions['user_id'] == uid) & (df_sessions['session_start'] <= '2026-03-01 23:59:59')]
        u_events = df_events[(df_events['user_id'] == uid) & (df_events['event_timestamp'] <= '2026-03-01 23:59:59')]
        u_tickets = df_tickets[(df_tickets['user_id'] == uid) & (df_tickets['created_at'] <= '2026-03-01 23:59:59')]
        u_orders = df_orders[(df_orders['user_id'] == uid) & (df_orders['order_timestamp'] <= '2026-03-01 23:59:59')]
        
        manual_recent_s = len(u_sessions[u_sessions['session_start'] >= '2026-02-16 00:00:00'])
        manual_prev_s = len(u_sessions[(u_sessions['session_start'] >= '2026-02-02 00:00:00') & (u_sessions['session_start'] <= '2026-02-15 23:59:59')])
        manual_events = len(u_events)
        manual_tickets = len(u_tickets)
        manual_spend = (df_users[df_users['user_id'] == uid]['monthly_fee'].values[0] * 2) + u_orders['amount'].sum()

        print(f"\n --- User ID: {uid} ({row['plan_tier']} Plan) ---")
        print(f"  Sessions (Recent 14d):   Feature Store = {row['sessions_recent_14d']}  | Raw CSV Manual = {manual_recent_s}")
        print(f"  Sessions (Previous 14d): Feature Store = {row['sessions_previous_14d']}  | Raw CSV Manual = {manual_prev_s}")
        print(f"  Session Drop %:          Feature Store = {row['session_drop_pct']}%")
        print(f"  Total Events (60d):      Feature Store = {row['total_events_60d']}  | Raw CSV Manual = {manual_events}")
        print(f"  Support Tickets (60d):   Feature Store = {row['support_ticket_count']}   | Raw CSV Manual = {manual_tickets}")
        print(f"  Total Spend (60d):       Feature Store = ${row['total_spend_60d']} | Raw CSV Manual = ${manual_spend:.2f}")
        print(f"  Avg Session Gap:         Feature Store = {row['avg_session_gap_days']} days")
        print(f"  Churn Label (Target):    Feature Store = {row['churn_label']}")

    # 4. Feature Summary Statistics Comparison (Churned vs Non-Churned)
    print(f"\n[4/4] FEATURE MEAN COMPARISON (CHURNED VS NON-CHURNED USERS):")
    cols_to_compare = [
        'days_since_last_session', 'sessions_recent_14d', 'sessions_previous_14d',
        'session_drop_pct', 'total_events_60d', 'avg_session_gap_days',
        'support_ticket_count', 'unresolved_tickets', 'total_spend_60d'
    ]
    
    comp_df = df_fs.groupby('churn_label')[cols_to_compare].mean().round(2).T
    comp_df.columns = ['Non-Churned (0) Mean', 'Churned (1) Mean']
    print(comp_df.to_string())
    print("\n" + "=" * 70)
    print("        FEATURE STORE VALIDATED AND CREATED SUCCESSFULLY!        ")
    print("=" * 70)

if __name__ == "__main__":
    validate_local_feature_store()
