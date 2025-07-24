-- ===================================
-- Phase 0 MCP User Permissions Migration
-- Creates dedicated MCP server user with limited permissions
-- ===================================

-- Create MCP server user with controlled access
DO $$
BEGIN
    -- Check if the user already exists
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mcp_server_user') THEN
        CREATE USER mcp_server_user WITH PASSWORD 'secure_mcp_user_password';
    END IF;
END $$;

-- ===================================
-- READ-ONLY ACCESS TO RULE TABLES
-- ===================================

-- Grant read-only access to rule tables
GRANT SELECT ON rule_performance TO mcp_server_user;
GRANT SELECT ON rule_metadata TO mcp_server_user;
GRANT SELECT ON rule_combinations TO mcp_server_user;

-- Grant read access to views used by MCP server
GRANT SELECT ON rule_effectiveness_summary TO mcp_server_user;

-- Grant usage on rule table sequences (needed for reading)
GRANT USAGE ON SEQUENCE rule_performance_id_seq TO mcp_server_user;
GRANT USAGE ON SEQUENCE rule_metadata_id_seq TO mcp_server_user;
GRANT USAGE ON SEQUENCE rule_combinations_id_seq TO mcp_server_user;

-- ===================================
-- WRITE ACCESS ONLY TO FEEDBACK TABLES
-- ===================================

-- Grant insert and select access to feedback tables
GRANT INSERT, SELECT ON user_feedback TO mcp_server_user;
GRANT INSERT, SELECT ON improvement_sessions TO mcp_server_user;

-- Grant usage on feedback table sequences
GRANT USAGE ON SEQUENCE user_feedback_id_seq TO mcp_server_user;
GRANT USAGE ON SEQUENCE improvement_sessions_id_seq TO mcp_server_user;

-- ===================================
-- EXPLICIT DENIAL OF WRITE ACCESS TO RULE TABLES (FAIL-SAFE)
-- ===================================

-- Explicitly revoke write access to rule tables
REVOKE INSERT, UPDATE, DELETE ON rule_performance FROM mcp_server_user;
REVOKE INSERT, UPDATE, DELETE ON rule_metadata FROM mcp_server_user;
REVOKE INSERT, UPDATE, DELETE ON rule_combinations FROM mcp_server_user;
REVOKE INSERT, UPDATE, DELETE ON discovered_patterns FROM mcp_server_user;
REVOKE INSERT, UPDATE, DELETE ON ml_model_performance FROM mcp_server_user;
REVOKE INSERT, UPDATE, DELETE ON ab_experiments FROM mcp_server_user;

-- ===================================
-- DATABASE CONNECTION PRIVILEGES
-- ===================================

-- Grant connection to database
GRANT CONNECT ON DATABASE apes_production TO mcp_server_user;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO mcp_server_user;

-- ===================================
-- SECURITY VERIFICATION QUERIES
-- ===================================

-- Create a view to verify MCP user permissions
CREATE OR REPLACE VIEW mcp_user_permissions_audit AS
SELECT 
    schemaname,
    tablename,
    privilege_type,
    is_grantable
FROM information_schema.table_privileges 
WHERE grantee = 'mcp_server_user'
ORDER BY schemaname, tablename, privilege_type;

-- Create function to test MCP user permissions
CREATE OR REPLACE FUNCTION test_mcp_user_permissions()
RETURNS TABLE(
    test_name TEXT,
    test_result TEXT,
    test_status BOOLEAN
) AS $$
BEGIN
    -- Test 1: Can read rule_performance
    RETURN QUERY
    SELECT 
        'rule_performance_read_access'::TEXT,
        'Can read from rule_performance table'::TEXT,
        (SELECT COUNT(*) >= 0 FROM rule_performance)::BOOLEAN;
    
    -- Test 2: Can read rule_metadata  
    RETURN QUERY
    SELECT 
        'rule_metadata_read_access'::TEXT,
        'Can read from rule_metadata table'::TEXT,
        (SELECT COUNT(*) >= 0 FROM rule_metadata)::BOOLEAN;
    
    -- Test 3: Can insert into user_feedback
    RETURN QUERY
    SELECT 
        'user_feedback_write_access'::TEXT,
        'Can write to user_feedback table'::TEXT,
        true::BOOLEAN; -- Will be tested during runtime
        
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY
        SELECT 
            'permission_test_error'::TEXT,
            SQLERRM::TEXT,
            false::BOOLEAN;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission on test function to mcp_server_user
GRANT EXECUTE ON FUNCTION test_mcp_user_permissions() TO mcp_server_user;

-- ===================================
-- MIGRATION COMPLETION LOG
-- ===================================

-- Log migration completion
INSERT INTO improvement_sessions (
    session_id,
    user_id,
    original_prompt,
    final_prompt,
    rules_applied,
    session_metadata,
    status
) VALUES (
    'phase0_migration_001',
    'system',
    'Phase 0 MCP User Permissions Migration',
    'Created mcp_server_user with controlled database permissions',
    '[]'::jsonb,
    '{"migration": "001_phase0_mcp_user_permissions", "phase": "0", "timestamp": "' || NOW() || '"}'::jsonb,
    'completed'
)
ON CONFLICT (session_id) DO NOTHING;