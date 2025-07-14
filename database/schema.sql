-- APES PostgreSQL Database Schema
-- Adaptive Prompt Enhancement System
-- Created: 2025-01-05

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- ===================================
-- Rule Performance Tracking Tables
-- ===================================

-- Track individual rule effectiveness
CREATE TABLE rule_performance (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(50) NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    prompt_id UUID DEFAULT uuid_generate_v4(),
    prompt_type VARCHAR(50),
    prompt_category VARCHAR(50),
    improvement_score FLOAT CHECK (improvement_score >= 0 AND improvement_score <= 1),
    confidence_level FLOAT CHECK (confidence_level >= 0 AND confidence_level <= 1),
    execution_time_ms INTEGER,
    rule_parameters JSONB,
    prompt_characteristics JSONB,
    before_metrics JSONB,
    after_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Track rule combinations and their effectiveness
CREATE TABLE rule_combinations (
    id SERIAL PRIMARY KEY,
    combination_id UUID DEFAULT uuid_generate_v4(),
    rule_set JSONB NOT NULL,
    prompt_type VARCHAR(50),
    combined_effectiveness FLOAT,
    individual_scores JSONB,
    sample_size INTEGER DEFAULT 1,
    statistical_confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ===================================
-- User Feedback and Interaction Tables
-- ===================================

-- Store user feedback on improved prompts
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    feedback_id UUID DEFAULT uuid_generate_v4(),
    original_prompt TEXT NOT NULL,
    improved_prompt TEXT NOT NULL,
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    applied_rules JSONB NOT NULL,
    user_context JSONB,
    improvement_areas JSONB, -- What specifically improved: clarity, specificity, etc.
    user_notes TEXT,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track prompt improvement sessions
CREATE TABLE improvement_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(50),
    original_prompt TEXT NOT NULL,
    final_prompt TEXT,
    rules_applied JSONB,
    iteration_count INTEGER DEFAULT 1,
    total_improvement_score FLOAT,
    session_metadata JSONB,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned'))
);

-- ===================================
-- ML Optimization and Analytics Tables
-- ===================================

-- Store ML model performance metrics
CREATE TABLE ml_model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'rule_selector', 'parameter_optimizer', 'pattern_detector'
    accuracy_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    training_data_size INTEGER,
    validation_data_size INTEGER,
    hyperparameters JSONB,
    feature_importance JSONB,
    model_artifacts_path TEXT,
    mlflow_run_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track discovered rule patterns
CREATE TABLE discovered_patterns (
    id SERIAL PRIMARY KEY,
    pattern_id UUID DEFAULT uuid_generate_v4(),
    pattern_name VARCHAR(100),
    pattern_description TEXT,
    pattern_rule JSONB NOT NULL,
    discovery_method VARCHAR(50), -- 'ml_mining', 'statistical_analysis', 'manual'
    effectiveness_score FLOAT,
    support_count INTEGER, -- How many times this pattern was effective
    confidence_interval JSONB,
    validation_status VARCHAR(20) DEFAULT 'pending' CHECK (validation_status IN ('pending', 'validated', 'rejected')),
    discovered_at TIMESTAMP DEFAULT NOW(),
    validated_at TIMESTAMP
);

-- ===================================
-- Configuration and Rule Management Tables
-- ===================================

-- Store rule configuration and metadata
CREATE TABLE rule_metadata (
    id SERIAL PRIMARY KEY,
    rule_id VARCHAR(50) UNIQUE NOT NULL,
    rule_name VARCHAR(100) NOT NULL,
    rule_category VARCHAR(50),
    rule_description TEXT,
    default_parameters JSONB,
    parameter_constraints JSONB,
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 100,
    rule_version VARCHAR(20) DEFAULT '1.0.0',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Track A/B testing experiments
CREATE TABLE ab_experiments (
    id SERIAL PRIMARY KEY,
    experiment_id UUID DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(100) NOT NULL,
    description TEXT,
    control_rules JSONB NOT NULL,
    treatment_rules JSONB NOT NULL,
    target_metric VARCHAR(50),
    sample_size_per_group INTEGER,
    current_sample_size INTEGER DEFAULT 0,
    significance_threshold FLOAT DEFAULT 0.05,
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('planning', 'running', 'completed', 'stopped')),
    results JSONB,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- ===================================
-- Indexes for Performance
-- ===================================

-- Rule performance indexes
CREATE INDEX idx_rule_performance_rule_id ON rule_performance(rule_id);
CREATE INDEX idx_rule_performance_prompt_type ON rule_performance(prompt_type);
CREATE INDEX idx_rule_performance_created_at ON rule_performance(created_at);
CREATE INDEX idx_rule_performance_improvement_score ON rule_performance(improvement_score);
CREATE INDEX idx_rule_performance_characteristics ON rule_performance USING GIN (prompt_characteristics);

-- User feedback indexes
CREATE INDEX idx_user_feedback_session_id ON user_feedback(session_id);
CREATE INDEX idx_user_feedback_rating ON user_feedback(user_rating);
CREATE INDEX idx_user_feedback_created_at ON user_feedback(created_at);
CREATE INDEX idx_user_feedback_applied_rules ON user_feedback USING GIN (applied_rules);

-- ML optimization indexes
CREATE INDEX idx_ml_model_performance_version ON ml_model_performance(model_version);
CREATE INDEX idx_ml_model_performance_type ON ml_model_performance(model_type);
CREATE INDEX idx_discovered_patterns_effectiveness ON discovered_patterns(effectiveness_score);
CREATE INDEX idx_discovered_patterns_status ON discovered_patterns(validation_status);

-- Session tracking indexes
CREATE INDEX idx_improvement_sessions_user_id ON improvement_sessions(user_id);
CREATE INDEX idx_improvement_sessions_status ON improvement_sessions(status);
CREATE INDEX idx_improvement_sessions_started_at ON improvement_sessions(started_at);

-- ===================================
-- Views for Common Queries
-- ===================================

-- Rule effectiveness summary view
CREATE VIEW rule_effectiveness_summary AS
SELECT 
    rule_id,
    rule_name,
    COUNT(*) as usage_count,
    AVG(improvement_score) as avg_improvement,
    STDDEV(improvement_score) as score_stddev,
    MIN(improvement_score) as min_improvement,
    MAX(improvement_score) as max_improvement,
    AVG(confidence_level) as avg_confidence,
    AVG(execution_time_ms) as avg_execution_time,
    COUNT(DISTINCT prompt_type) as prompt_types_count
FROM rule_performance 
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY rule_id, rule_name
ORDER BY avg_improvement DESC;

-- User satisfaction summary view
CREATE VIEW user_satisfaction_summary AS
SELECT 
    DATE_TRUNC('day', created_at) as feedback_date,
    COUNT(*) as total_feedback,
    AVG(user_rating::FLOAT) as avg_rating,
    COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as positive_feedback,
    COUNT(CASE WHEN user_rating <= 2 THEN 1 END) as negative_feedback,
    ARRAY_AGG(DISTINCT (applied_rules->>'rule_id')) as rules_used
FROM user_feedback 
WHERE created_at >= NOW() - INTERVAL '90 days'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY feedback_date DESC;

-- ===================================
-- Triggers for Automatic Updates
-- ===================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_rule_performance_updated_at BEFORE UPDATE ON rule_performance
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rule_combinations_updated_at BEFORE UPDATE ON rule_combinations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rule_metadata_updated_at BEFORE UPDATE ON rule_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===================================
-- Initial Data and Configuration
-- ===================================

-- Insert default rule metadata (based on existing rules in your system)
INSERT INTO rule_metadata (rule_id, rule_name, rule_category, rule_description, default_parameters, priority) VALUES
('clarity_rule', 'Clarity Enhancement Rule', 'core', 'Improves prompt clarity and reduces ambiguity', '{"min_improvement_threshold": 0.1}', 100),
('specificity_rule', 'Specificity Enhancement Rule', 'core', 'Makes prompts more specific and actionable', '{"context_expansion": true}', 90),
('structure_rule', 'Structure Enhancement Rule', 'formatting', 'Improves prompt structure and organization', '{"add_formatting": true}', 80),
('context_rule', 'Context Enhancement Rule', 'content', 'Adds relevant context and background information', '{"context_depth": "medium"}', 70);

-- Create database user for the application (optional)
-- This would typically be done in init.sql