import os
import uuid
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_synthetic_dataset(num_users=3000, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic dataset for {num_users} users over 90 days...")

    start_date = datetime(2026, 1, 1, 0, 0, 0)
    observation_cutoff = datetime(2026, 3, 1, 23, 59, 59) # End of Day 60
    end_date = datetime(2026, 3, 31, 23, 59, 59)           # End of Day 90

    # 1. Generate Users (All signed up before Jan 1, 2026)
    plan_tiers = ['Basic', 'Pro', 'Enterprise']
    plan_weights = [0.50, 0.35, 0.15]
    fee_map = {'Basic': 29.00, 'Pro': 99.00, 'Enterprise': 299.00}

    users = []
    user_cohorts = {} # user_id -> cohort_type

    for uid in range(1001, 1001 + num_users):
        signup_dt = datetime(2025, 12, 1) + timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        plan = np.random.choice(plan_tiers, p=plan_weights)
        fee = fee_map[plan]
        
        # Assign latent cohort type
        cohort_p = np.random.choice(['engaged', 'gradual_decline', 'friction'], p=[0.50, 0.33, 0.17])
        user_cohorts[uid] = cohort_p

        users.append({
            'user_id': uid,
            'signup_date': signup_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'plan_tier': plan,
            'monthly_fee': fee
        })

    df_users = pd.DataFrame(users)

    # 2. Generate Sessions, Events, Orders, and Support Tickets over 90 days
    sessions = []
    events = []
    orders = []
    support_tickets = []

    devices = ['Desktop', 'Mobile', 'Tablet']
    event_types = ['page_view', 'feature_used', 'export_data', 'settings_update']
    event_weights = [0.55, 0.25, 0.12, 0.08]

    ticket_categories = ['billing', 'bug', 'feature_request', 'onboarding']
    
    session_id_counter = 1
    event_id_counter = 1
    order_id_counter = 1
    ticket_id_counter = 1

    for uid in range(1001, 1001 + num_users):
        cohort = user_cohorts[uid]
        plan = df_users.loc[df_users['user_id'] == uid, 'plan_tier'].values[0]
        fee = fee_map[plan]

        # Generate friction ticket day if in friction cohort
        friction_day = random.randint(30, 48) if cohort == 'friction' else None

        for day_idx in range(90):
            current_day = start_date + timedelta(days=day_idx)

            # Calculate daily session probability based on cohort and decay
            if cohort == 'engaged':
                p_session = 0.65 + np.random.normal(0, 0.05)
            elif cohort == 'gradual_decline':
                # Decay session probability over 90 days
                p_session = max(0.02, 0.60 * (1 - (day_idx / 90.0) ** 0.8) + np.random.normal(0, 0.04))
            elif cohort == 'friction':
                if day_idx < friction_day:
                    p_session = 0.55 + np.random.normal(0, 0.05)
                else:
                    p_session = max(0.01, 0.08 * (0.85 ** (day_idx - friction_day)) + np.random.normal(0, 0.02))

            p_session = max(0.0, min(1.0, p_session))

            # Simulate sessions for this day
            if random.random() < p_session:
                # 1 to 2 sessions per active day
                num_sessions = 1 if random.random() > 0.20 else 2
                for s in range(num_sessions):
                    s_id = f"s_{session_id_counter}"
                    session_id_counter += 1

                    s_start = current_day + timedelta(hours=random.randint(6, 22), minutes=random.randint(0, 59))
                    duration_mins = random.randint(3, 45)
                    s_end = s_start + timedelta(minutes=duration_mins)
                    device = np.random.choice(devices, p=[0.60, 0.30, 0.10])

                    sessions.append({
                        'session_id': s_id,
                        'user_id': uid,
                        'session_start': s_start.strftime('%Y-%m-%d %H:%M:%S'),
                        'session_end': s_end.strftime('%Y-%m-%d %H:%M:%S'),
                        'device_type': device
                    })

                    # Generate events inside this session
                    num_events = random.randint(2, 7)
                    for e in range(num_events):
                        e_id = f"e_{event_id_counter}"
                        event_id_counter += 1
                        e_time = s_start + timedelta(minutes=random.randint(0, max(1, duration_mins - 1)))
                        ename = np.random.choice(event_types, p=event_weights)

                        events.append({
                            'event_id': e_id,
                            'session_id': s_id,
                            'user_id': uid,
                            'event_timestamp': e_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'event_name': ename
                        })

            # Add-on orders generation (probabilistic)
            if p_session > 0.20 and random.random() < 0.008:
                o_id = f"o_{order_id_counter}"
                order_id_counter += 1
                o_time = current_day + timedelta(hours=random.randint(9, 18))
                amt = random.choice([19.00, 49.00, 99.00, 149.00])
                orders.append({
                    'order_id': o_id,
                    'user_id': uid,
                    'order_timestamp': o_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'amount': amt
                })

            # Support tickets generation
            ticket_prob = 0.002
            if cohort == 'friction' and day_idx == friction_day:
                ticket_prob = 0.90 # High ticket prob on friction day

            if random.random() < ticket_prob:
                t_id = f"t_{ticket_id_counter}"
                ticket_id_counter += 1
                t_time = current_day + timedelta(hours=random.randint(8, 17))
                cat = np.random.choice(ticket_categories, p=[0.35, 0.35, 0.15, 0.15])
                
                # Friction tickets stay open/escalated more often
                if cohort == 'friction' and day_idx >= friction_day:
                    status = np.random.choice(['open', 'escalated', 'resolved'], p=[0.45, 0.40, 0.15])
                else:
                    status = np.random.choice(['resolved', 'open'], p=[0.85, 0.15])

                support_tickets.append({
                    'ticket_id': t_id,
                    'user_id': uid,
                    'created_at': t_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'category': cat,
                    'status': status
                })

    df_sessions = pd.DataFrame(sessions)
    df_events = pd.DataFrame(events)
    df_orders = pd.DataFrame(orders)
    df_tickets = pd.DataFrame(support_tickets)

    # Save to CSV
    df_users.to_csv(os.path.join(output_dir, "users.csv"), index=False)
    df_sessions.to_csv(os.path.join(output_dir, "sessions.csv"), index=False)
    df_events.to_csv(os.path.join(output_dir, "events.csv"), index=False)
    df_orders.to_csv(os.path.join(output_dir, "orders.csv"), index=False)
    df_tickets.to_csv(os.path.join(output_dir, "support_tickets.csv"), index=False)

    print(f"Dataset generated successfully in '{output_dir}/':")
    print(f" - Users: {len(df_users)}")
    print(f" - Sessions: {len(df_sessions)}")
    print(f" - Events: {len(df_events)}")
    print(f" - Orders: {len(df_orders)}")
    print(f" - Support Tickets: {len(df_tickets)}")

    # Calculate Churn Statistics for verification
    future_start = "2026-03-02 00:00:00"
    active_future_users = set(df_sessions[df_sessions['session_start'] >= future_start]['user_id']).union(
        set(df_orders[df_orders['order_timestamp'] >= future_start]['user_id'])
    )
    
    all_users_set = set(df_users['user_id'])
    churned_users = all_users_set - active_future_users

    total_users = len(df_users)
    churn_count = len(churned_users)
    non_churn_count = total_users - churn_count
    churn_rate = (churn_count / total_users) * 100

    print("\n--- Emergent Churn Label Summary (Days 61-90 Activity) ---")
    print(f" Total Users:      {total_users}")
    print(f" Churned Users:    {churn_count}")
    print(f" Non-Churned:      {non_churn_count}")
    print(f" Churn Rate:       {churn_rate:.2f}%")

if __name__ == "__main__":
    generate_synthetic_dataset()
