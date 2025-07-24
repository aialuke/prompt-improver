-- PostgreSQL Replication Setup Script
-- Creates replication user and configures streaming replication

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'replication_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE apes_production TO replicator;

-- Create replication slot for streaming replication
SELECT pg_create_physical_replication_slot('replica_slot');

-- Configure pg_hba.conf for replication (this would typically be done via configuration)
-- host replication replicator 0.0.0.0/0 md5

-- Create monitoring functions for replication status
CREATE OR REPLACE FUNCTION get_replication_status()
RETURNS TABLE(
    slot_name text,
    active boolean,
    restart_lsn pg_lsn,
    confirmed_flush_lsn pg_lsn,
    lag_bytes bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rs.slot_name::text,
        rs.active,
        rs.restart_lsn,
        rs.confirmed_flush_lsn,
        CASE 
            WHEN rs.confirmed_flush_lsn IS NOT NULL 
            THEN pg_wal_lsn_diff(pg_current_wal_lsn(), rs.confirmed_flush_lsn)
            ELSE NULL
        END as lag_bytes
    FROM pg_replication_slots rs
    WHERE rs.slot_type = 'physical';
END;
$$ LANGUAGE plpgsql;

-- Create function to check if this is the primary
CREATE OR REPLACE FUNCTION is_primary()
RETURNS boolean AS $$
BEGIN
    RETURN NOT pg_is_in_recovery();
END;
$$ LANGUAGE plpgsql;

-- Create function to get replication lag
CREATE OR REPLACE FUNCTION get_replication_lag()
RETURNS interval AS $$
DECLARE
    lag_seconds numeric;
BEGIN
    IF pg_is_in_recovery() THEN
        -- This is a replica, get lag from primary
        SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) INTO lag_seconds;
        RETURN make_interval(secs => COALESCE(lag_seconds, 0));
    ELSE
        -- This is primary, no lag
        RETURN interval '0 seconds';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create health check function
CREATE OR REPLACE FUNCTION health_check()
RETURNS json AS $$
DECLARE
    result json;
    is_primary_db boolean;
    replication_lag interval;
    active_connections integer;
BEGIN
    -- Get basic health information
    SELECT is_primary() INTO is_primary_db;
    SELECT get_replication_lag() INTO replication_lag;
    SELECT count(*) FROM pg_stat_activity WHERE state = 'active' INTO active_connections;
    
    -- Build result JSON
    SELECT json_build_object(
        'timestamp', now(),
        'is_primary', is_primary_db,
        'replication_lag_seconds', EXTRACT(EPOCH FROM replication_lag),
        'active_connections', active_connections,
        'database_size_mb', pg_database_size(current_database()) / 1024 / 1024,
        'uptime_seconds', EXTRACT(EPOCH FROM (now() - pg_postmaster_start_time())),
        'version', version()
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permissions on monitoring functions
GRANT EXECUTE ON FUNCTION get_replication_status() TO apes_user;
GRANT EXECUTE ON FUNCTION is_primary() TO apes_user;
GRANT EXECUTE ON FUNCTION get_replication_lag() TO apes_user;
GRANT EXECUTE ON FUNCTION health_check() TO apes_user;
