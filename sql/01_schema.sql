-- 01_schema.sql: PostgreSQL DDL Script for ChurnGuard AI
-- Creates raw relational tables and appropriate indexes for feature store performance.

DROP TABLE IF EXISTS events CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. USERS TABLE
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    signup_date TIMESTAMP NOT NULL,
    plan_tier VARCHAR(20) NOT NULL,
    monthly_fee NUMERIC(8, 2) NOT NULL
);

-- 2. SESSIONS TABLE
CREATE TABLE sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP NOT NULL,
    device_type VARCHAR(20) NOT NULL
);

-- 3. EVENTS TABLE
CREATE TABLE events (
    event_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    event_timestamp TIMESTAMP NOT NULL,
    event_name VARCHAR(50) NOT NULL
);

-- 4. ORDERS TABLE
CREATE TABLE orders (
    order_id VARCHAR(36) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    order_timestamp TIMESTAMP NOT NULL,
    amount NUMERIC(8, 2) NOT NULL
);

-- 5. SUPPORT TICKETS TABLE
CREATE TABLE support_tickets (
    ticket_id VARCHAR(36) PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP NOT NULL,
    category VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL
);

-- INDEXES FOR FAST TEMPORAL AGGREGATIONS
CREATE INDEX idx_sessions_user_time ON sessions(user_id, session_start);
CREATE INDEX idx_events_user_time ON events(user_id, event_timestamp);
CREATE INDEX idx_orders_user_time ON orders(user_id, order_timestamp);
CREATE INDEX idx_tickets_user_time ON support_tickets(user_id, created_at);
