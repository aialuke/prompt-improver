-- ===================================
-- Phase 0 Unified Feedback Schema Migration
-- Creates prompt_improvement_sessions table for MCP server feedback collection
-- ===================================

-- Create unified feedback table for MCP server
CREATE TABLE IF NOT EXISTS prompt_improvement_sessions (
    id SERIAL PRIMARY KEY,
    original_prompt TEXT NOT NULL,
    enhanced_prompt TEXT NOT NULL,
    applied_rules JSONB NOT NULL,
    response_time_ms INTEGER NOT NULL,
    cache_hit_level VARCHAR(10), -- 'L1', 'L2', 'L3', 'MISS'
    agent_type VARCHAR(50), -- 'claude-code', 'augment-code', 'external-agent'
    session_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    anonymized_user_hash VARCHAR(64), -- SHA-256 hash for privacy
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_cache_hit_level CHECK (cache_hit_level IN ('L1', 'L2', 'L3', 'MISS', NULL)),
    CONSTRAINT valid_agent_type CHECK (agent_type IN ('claude-code', 'augment-code', 'external-agent', NULL)),
    CONSTRAINT positive_response_time CHECK (response_time_ms > 0),
    CONSTRAINT reasonable_response_time CHECK (response_time_ms < 30000), -- 30 seconds max
    CONSTRAINT non_empty_prompt CHECK (LENGTH(TRIM(original_prompt)) > 0),
    CONSTRAINT non_empty_enhanced CHECK (LENGTH(TRIM(enhanced_prompt)) > 0)
);

-- ===================================
-- PERFORMANCE INDEXES
-- ===================================

-- Primary performance indexes
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON prompt_improvement_sessions(session_timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_rules ON prompt_improvement_sessions USING GIN(applied_rules);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_type ON prompt_improvement_sessions(agent_type);
CREATE INDEX IF NOT EXISTS idx_sessions_cache_level ON prompt_improvement_sessions(cache_hit_level);
CREATE INDEX IF NOT EXISTS idx_sessions_response_time ON prompt_improvement_sessions(response_time_ms);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON prompt_improvement_sessions(created_at);

-- Composite indexes for analytics
CREATE INDEX IF NOT EXISTS idx_sessions_agent_timestamp ON prompt_improvement_sessions(agent_type, session_timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_cache_response_time ON prompt_improvement_sessions(cache_hit_level, response_time_ms);

-- ===================================
-- GRANT PERMISSIONS TO MCP USER
-- ===================================

-- Grant write access to MCP server user
GRANT INSERT, SELECT ON prompt_improvement_sessions TO mcp_server_user;
GRANT USAGE ON SEQUENCE prompt_improvement_sessions_id_seq TO mcp_server_user;

-- ===================================
-- ANALYTICS VIEWS
-- ===================================

-- Real-time performance metrics view
CREATE OR REPLACE VIEW mcp_performance_metrics AS
SELECT 
    agent_type,
    cache_hit_level,
    COUNT(*) as session_count,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response_time,
    MIN(response_time_ms) as min_response_time,
    MAX(response_time_ms) as max_response_time,
    STDDEV(response_time_ms) as response_time_stddev,
    COUNT(CASE WHEN response_time_ms <= 200 THEN 1 END) as under_200ms_count,
    ROUND(
        (COUNT(CASE WHEN response_time_ms <= 200 THEN 1 END)::FLOAT / COUNT(*)) * 100, 
        2
    ) as sla_compliance_percentage
FROM prompt_improvement_sessions
WHERE session_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY agent_type, cache_hit_level
ORDER BY agent_type, cache_hit_level;

-- Cache effectiveness analysis view
CREATE OR REPLACE VIEW mcp_cache_effectiveness AS
SELECT 
    cache_hit_level,
    COUNT(*) as hit_count,
    ROUND(
        (COUNT(*)::FLOAT / (SELECT COUNT(*) FROM prompt_improvement_sessions WHERE session_timestamp >= NOW() - INTERVAL '24 hours')) * 100, 
        2
    ) as hit_percentage,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
FROM prompt_improvement_sessions
WHERE session_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY cache_hit_level
ORDER BY 
    CASE cache_hit_level 
        WHEN 'L1' THEN 1 
        WHEN 'L2' THEN 2 
        WHEN 'L3' THEN 3 
        WHEN 'MISS' THEN 4 
        ELSE 5 
    END;

-- Rule usage analytics view
CREATE OR REPLACE VIEW mcp_rule_usage_analytics AS
WITH rule_extracts AS (
    SELECT 
        jsonb_array_elements(applied_rules) as rule_data,
        response_time_ms,
        session_timestamp,
        agent_type
    FROM prompt_improvement_sessions
    WHERE session_timestamp >= NOW() - INTERVAL '7 days'
)
SELECT 
    rule_data->>'rule_id' as rule_id,
    rule_data->>'rule_name' as rule_name,
    COUNT(*) as usage_count,
    COUNT(DISTINCT agent_type) as agent_types_using,
    AVG(response_time_ms) as avg_response_time,
    ROUND(
        (COUNT(*)::FLOAT / (SELECT COUNT(*) FROM prompt_improvement_sessions WHERE session_timestamp >= NOW() - INTERVAL '7 days')) * 100, 
        2
    ) as usage_percentage
FROM rule_extracts
WHERE rule_data->>'rule_id' IS NOT NULL
GROUP BY rule_data->>'rule_id', rule_data->>'rule_name'
ORDER BY usage_count DESC;

-- Hourly performance trends view
CREATE OR REPLACE VIEW mcp_hourly_performance_trends AS
SELECT 
    DATE_TRUNC('hour', session_timestamp) as hour_bucket,
    COUNT(*) as total_sessions,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    COUNT(CASE WHEN response_time_ms <= 200 THEN 1 END) as sla_compliant_sessions,
    ROUND(
        (COUNT(CASE WHEN response_time_ms <= 200 THEN 1 END)::FLOAT / COUNT(*)) * 100, 
        2
    ) as sla_compliance_percentage,
    COUNT(CASE WHEN cache_hit_level != 'MISS' THEN 1 END) as cache_hits,
    ROUND(
        (COUNT(CASE WHEN cache_hit_level != 'MISS' THEN 1 END)::FLOAT / COUNT(*)) * 100, 
        2
    ) as cache_hit_percentage
FROM prompt_improvement_sessions
WHERE session_timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', session_timestamp)
ORDER BY hour_bucket DESC;

-- ===================================
-- GRANT VIEW ACCESS TO MCP USER
-- ===================================

-- Grant read access to analytics views for MCP server monitoring
GRANT SELECT ON mcp_performance_metrics TO mcp_server_user;
GRANT SELECT ON mcp_cache_effectiveness TO mcp_server_user;
GRANT SELECT ON mcp_rule_usage_analytics TO mcp_server_user;
GRANT SELECT ON mcp_hourly_performance_trends TO mcp_server_user;

-- ===================================
-- DATA RETENTION AND CLEANUP
-- ===================================

-- Create function for data retention cleanup
CREATE OR REPLACE FUNCTION cleanup_old_prompt_sessions(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM prompt_improvement_sessions 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log cleanup operation
    INSERT INTO improvement_sessions (
        session_id,
        user_id,
        original_prompt,
        final_prompt,
        rules_applied,
        session_metadata,
        status
    ) VALUES (
        'cleanup_' || extract(epoch from NOW())::text,
        'system',
        'Automated cleanup of old prompt improvement sessions',
        'Deleted ' || deleted_count || ' sessions older than ' || retention_days || ' days',
        '[]'::jsonb,
        '{"cleanup": true, "deleted_count": ' || deleted_count || ', "retention_days": ' || retention_days || ', "timestamp": "' || NOW() || '"}'::jsonb,
        'completed'
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ===================================
-- TESTING AND VALIDATION FUNCTIONS
-- ===================================

-- Create function to test MCP feedback collection
CREATE OR REPLACE FUNCTION test_mcp_feedback_collection()
RETURNS TABLE(
    test_name TEXT,
    test_result TEXT,
    test_status BOOLEAN
) AS $$
BEGIN
    -- Test 1: Insert test feedback record
    BEGIN
        INSERT INTO prompt_improvement_sessions (
            original_prompt,
            enhanced_prompt,
            applied_rules,
            response_time_ms,
            cache_hit_level,
            agent_type,
            anonymized_user_hash
        ) VALUES (
            'Test prompt for validation',
            'Enhanced test prompt for validation',
            '[{"rule_id": "test_rule", "rule_name": "Test Rule"}]'::jsonb,
            150,
            'L1',
            'claude-code',
            'test_hash_' || extract(epoch from NOW())::text
        );
        
        RETURN QUERY
        SELECT 
            'feedback_insert_test'::TEXT,
            'Successfully inserted test feedback record'::TEXT,
            true::BOOLEAN;
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY
        SELECT 
            'feedback_insert_test'::TEXT,
            'Failed to insert test feedback: ' || SQLERRM::TEXT,
            false::BOOLEAN;
    END;
    
    -- Test 2: Verify analytics views work
    BEGIN
        PERFORM COUNT(*) FROM mcp_performance_metrics;
        
        RETURN QUERY
        SELECT 
            'analytics_views_test'::TEXT,
            'Analytics views are accessible'::TEXT,
            true::BOOLEAN;
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY
        SELECT 
            'analytics_views_test'::TEXT,
            'Analytics views error: ' || SQLERRM::TEXT,
            false::BOOLEAN;
    END;
    
    -- Test 3: Verify constraints work
    BEGIN
        INSERT INTO prompt_improvement_sessions (
            original_prompt,
            enhanced_prompt,
            applied_rules,
            response_time_ms,
            cache_hit_level
        ) VALUES (
            '',  -- Empty prompt should fail
            'Enhanced test prompt',
            '[]'::jsonb,
            150,
            'L1'
        );
        
        RETURN QUERY
        SELECT 
            'constraint_test'::TEXT,
            'Constraint validation failed - empty prompt was accepted'::TEXT,
            false::BOOLEAN;
    EXCEPTION WHEN OTHERS THEN
        RETURN QUERY
        SELECT 
            'constraint_test'::TEXT,
            'Constraints working correctly - empty prompt rejected'::TEXT,
            true::BOOLEAN;
    END;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on test function to mcp_server_user
GRANT EXECUTE ON FUNCTION test_mcp_feedback_collection() TO mcp_server_user;

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
    'phase0_migration_002',
    'system',
    'Phase 0 Unified Feedback Schema Migration',
    'Created prompt_improvement_sessions table with analytics views and permissions',
    '[]'::jsonb,
    '{"migration": "002_phase0_unified_feedback_schema", "phase": "0", "timestamp": "' || NOW() || '"}'::jsonb,
    'completed'
)
ON CONFLICT (session_id) DO NOTHING;