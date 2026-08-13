import os
import sys
import argparse
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pandas as pd

def load_data_to_postgres(host="localhost", port=5432, dbname="churnguard_db", user="postgres", password="postgres"):
    print(f"Connecting to PostgreSQL server at {host}:{port} as user '{user}'...")
    
    # 1. Connect to default 'postgres' database to ensure target db exists
    try:
        conn_default = psycopg2.connect(
            host=host, port=port, dbname="postgres", user=user, password=password
        )
        conn_default.autocommit = True
        cur_default = conn_default.cursor()

        # Check if churnguard_db exists
        cur_default.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
        exists = cur_default.fetchone()
        if not exists:
            print(f"Database '{dbname}' does not exist. Creating it now...")
            cur_default.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname)))
        else:
            print(f"Database '{dbname}' already exists.")

        cur_default.close()
        conn_default.close()

    except Exception as e:
        print(f"\n[ERROR] Failed to connect to PostgreSQL default database: {e}")
        print("Please verify PostgreSQL is running locally and credentials (host, port, user, password) are correct.")
        sys.exit(1)

    # 2. Connect to churnguard_db and execute schema DDL
    print(f"Connecting to target database '{dbname}'...")
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    )
    cur = conn.cursor()

    schema_file = os.path.join("sql", "01_schema.sql")
    if os.path.exists(schema_file):
        print(f"Executing schema DDL script '{schema_file}'...")
        with open(schema_file, 'r') as f:
            ddl_sql = f.read()
        cur.execute(ddl_sql)
        conn.commit()
    else:
        print(f"[ERROR] Schema DDL file '{schema_file}' not found!")
        sys.exit(1)

    # 3. Load CSV files into PostgreSQL tables using bulk insertion
    csv_table_map = [
        ("users.csv", "users", ["user_id", "signup_date", "plan_tier", "monthly_fee"]),
        ("sessions.csv", "sessions", ["session_id", "user_id", "session_start", "session_end", "device_type"]),
        ("events.csv", "events", ["event_id", "session_id", "user_id", "event_timestamp", "event_name"]),
        ("orders.csv", "orders", ["order_id", "user_id", "order_timestamp", "amount"]),
        ("support_tickets.csv", "support_tickets", ["ticket_id", "user_id", "created_at", "category", "status"])
    ]

    for csv_file, table_name, columns in csv_table_map:
        csv_path = os.path.join("data", csv_file)
        if not os.path.exists(csv_path):
            print(f"[WARNING] File {csv_path} not found. Skipping table {table_name}.")
            continue

        print(f"Bulk loading '{csv_path}' into PostgreSQL table '{table_name}'...")
        df = pd.read_csv(csv_path)

        cols_str = ", ".join(columns)
        query = f"INSERT INTO {table_name} ({cols_str}) VALUES %s"

        tuples = [tuple(x) for x in df[columns].to_numpy()]
        execute_values(cur, query, tuples, page_size=5000)
        conn.commit()

        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cur.fetchone()[0]
        print(f" -> Table '{table_name}' loaded successfully with {row_count} rows.")

    cur.close()
    conn.close()
    print("\n[SUCCESS] All tables loaded into PostgreSQL successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load ChurnGuard AI CSV data into PostgreSQL")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--dbname", default="churnguard_db", help="PostgreSQL database name")
    parser.add_argument("--user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--password", default="postgres", help="PostgreSQL password")
    
    args = parser.parse_args()
    load_data_to_postgres(args.host, args.port, args.dbname, args.user, args.password)
