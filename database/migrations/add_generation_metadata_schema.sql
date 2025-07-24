-- ===================================
-- Generation Metadata Schema Enhancement (Week 6)
-- Adds comprehensive tracking for synthetic data generation
-- ===================================

-- Generation sessions table for tracking generation workflows
CREATE TABLE IF NOT EXISTS generation_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    session_type VARCHAR(50) NOT NULL DEFAULT 'synthetic_data', -- 'synthetic_data', 'targeted_generation', 'hybrid_generation'
    training_session_id VARCHAR(100), -- Link to training sessions if applicable
    
    -- Generation configuration
    generation_method VARCHAR(50) NOT NULL, -- 'statistical', 'neural', 'hybrid', 'diffusion'
    target_samples INTEGER NOT NULL,
    batch_size INTEGER,
    quality_threshold FLOAT DEFAULT 0.7,
    performance_gaps JSONB, -- Performance gaps that triggered generation
    focus_areas JSONB, -- Areas of focus for generation
    
    -- Session status and timing
    status VARCHAR(20) DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    total_duration_seconds FLOAT,
    
    -- Results summary
    samples_generated INTEGER DEFAULT 0,
    samples_filtered INTEGER DEFAULT 0,
    final_sample_count INTEGER DEFAULT 0,
    average_quality_score FLOAT,
    generation_efficiency FLOAT,
    
    -- Metadata
    configuration JSONB, -- Full generation configuration
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Generation batches table for tracking individual batch processing
CREATE TABLE IF NOT EXISTS generation_batches (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) UNIQUE NOT NULL,
    session_id VARCHAR(100) REFERENCES generation_sessions(session_id) ON DELETE CASCADE,
    
    -- Batch configuration
    batch_number INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    generation_method VARCHAR(50) NOT NULL,
    
    -- Performance metrics
    processing_time_seconds FLOAT,
    memory_usage_mb FLOAT,
    memory_peak_mb FLOAT,
    throughput_samples_per_sec FLOAT,
    efficiency_score FLOAT,
    success_rate FLOAT,
    
    -- Results
    samples_requested INTEGER,
    samples_generated INTEGER,
    samples_filtered INTEGER,
    error_count INTEGER DEFAULT 0,
    
    -- Quality metrics
    average_quality_score FLOAT,
    quality_score_range JSONB, -- [min, max] quality scores
    diversity_score FLOAT,
    
    -- Metadata
    batch_metadata JSONB,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(session_id, batch_number)
);

-- Method performance tracking for auto-selection
CREATE TABLE IF NOT EXISTS generation_method_performance (
    id SERIAL PRIMARY KEY,
    method_name VARCHAR(50) NOT NULL,
    session_id VARCHAR(100) REFERENCES generation_sessions(session_id) ON DELETE CASCADE,
    
    -- Performance metrics
    generation_time_seconds FLOAT NOT NULL,
    quality_score FLOAT NOT NULL,
    diversity_score FLOAT NOT NULL,
    memory_usage_mb FLOAT NOT NULL,
    success_rate FLOAT NOT NULL,
    samples_generated INTEGER NOT NULL,
    
    -- Context
    performance_gaps_addressed JSONB,
    batch_size INTEGER,
    configuration JSONB,
    
    -- Tracking
    recorded_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_method_performance_method (method_name),
    INDEX idx_method_performance_session (session_id),
    INDEX idx_method_performance_recorded (recorded_at)
);

-- Synthetic data samples table for storing generated data
CREATE TABLE IF NOT EXISTS synthetic_data_samples (
    id SERIAL PRIMARY KEY,
    sample_id VARCHAR(100) UNIQUE NOT NULL,
    session_id VARCHAR(100) REFERENCES generation_sessions(session_id) ON DELETE CASCADE,
    batch_id VARCHAR(100) REFERENCES generation_batches(batch_id) ON DELETE CASCADE,
    
    -- Sample data
    feature_vector JSONB NOT NULL, -- The actual feature data
    effectiveness_score FLOAT,
    quality_score FLOAT,
    
    -- Classification
    domain_category VARCHAR(50), -- 'technical', 'creative', 'analytical', etc.
    difficulty_level VARCHAR(20), -- 'easy', 'medium', 'hard'
    focus_areas JSONB, -- Areas this sample targets
    
    -- Generation metadata
    generation_method VARCHAR(50),
    generation_strategy VARCHAR(50),
    targeting_info JSONB,
    
    -- Lifecycle
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'filtered', 'archived', 'deleted')),
    created_at TIMESTAMP DEFAULT NOW(),
    archived_at TIMESTAMP,
    
    INDEX idx_synthetic_samples_session (session_id),
    INDEX idx_synthetic_samples_batch (batch_id),
    INDEX idx_synthetic_samples_domain (domain_category),
    INDEX idx_synthetic_samples_quality (quality_score),
    INDEX idx_synthetic_samples_status (status)
);

-- Quality assessment results table
CREATE TABLE IF NOT EXISTS generation_quality_assessments (
    id SERIAL PRIMARY KEY,
    assessment_id VARCHAR(100) UNIQUE NOT NULL,
    session_id VARCHAR(100) REFERENCES generation_sessions(session_id) ON DELETE CASCADE,
    batch_id VARCHAR(100) REFERENCES generation_batches(batch_id) ON DELETE CASCADE,
    
    -- Assessment configuration
    assessment_type VARCHAR(50) NOT NULL, -- 'multi_dimensional', 'distribution_validation', 'outlier_detection'
    quality_threshold FLOAT,
    
    -- Results
    overall_quality_score FLOAT NOT NULL,
    feature_validity_score FLOAT,
    diversity_score FLOAT,
    correlation_score FLOAT,
    distribution_score FLOAT,
    
    -- Detailed metrics
    samples_assessed INTEGER,
    samples_passed INTEGER,
    samples_failed INTEGER,
    outliers_detected INTEGER,
    
    -- Assessment details
    assessment_results JSONB, -- Detailed assessment results
    recommendations JSONB, -- Quality improvement recommendations
    
    -- Metadata
    assessed_at TIMESTAMP DEFAULT NOW(),
    assessment_duration_seconds FLOAT,
    
    INDEX idx_quality_assessments_session (session_id),
    INDEX idx_quality_assessments_type (assessment_type),
    INDEX idx_quality_assessments_score (overall_quality_score)
);

-- Generation analytics and trends table
CREATE TABLE IF NOT EXISTS generation_analytics (
    id SERIAL PRIMARY KEY,
    analytics_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Time period
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly'
    
    -- Aggregated metrics
    total_sessions INTEGER DEFAULT 0,
    total_samples_generated INTEGER DEFAULT 0,
    total_generation_time_seconds FLOAT DEFAULT 0,
    average_quality_score FLOAT,
    average_efficiency_score FLOAT,
    
    -- Method performance
    method_performance_summary JSONB, -- Performance by method
    best_performing_method VARCHAR(50),
    worst_performing_method VARCHAR(50),
    
    -- Trends
    quality_trend FLOAT, -- -1 to 1, negative = declining
    efficiency_trend FLOAT,
    volume_trend FLOAT,
    
    -- Resource usage
    total_memory_usage_mb FLOAT,
    peak_memory_usage_mb FLOAT,
    average_batch_size FLOAT,
    
    -- Metadata
    calculated_at TIMESTAMP DEFAULT NOW(),
    calculation_duration_seconds FLOAT,
    
    INDEX idx_generation_analytics_period (period_start, period_end),
    INDEX idx_generation_analytics_type (period_type),
    INDEX idx_generation_analytics_calculated (calculated_at)
);

-- ===================================
-- Indexes for Performance Optimization
-- ===================================

-- Session-based queries
CREATE INDEX IF NOT EXISTS idx_generation_sessions_status ON generation_sessions(status);
CREATE INDEX IF NOT EXISTS idx_generation_sessions_method ON generation_sessions(generation_method);
CREATE INDEX IF NOT EXISTS idx_generation_sessions_started ON generation_sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_generation_sessions_training ON generation_sessions(training_session_id);

-- Batch performance queries
CREATE INDEX IF NOT EXISTS idx_generation_batches_session_batch ON generation_batches(session_id, batch_number);
CREATE INDEX IF NOT EXISTS idx_generation_batches_efficiency ON generation_batches(efficiency_score);
CREATE INDEX IF NOT EXISTS idx_generation_batches_created ON generation_batches(created_at);

-- Method performance queries for auto-selection
CREATE INDEX IF NOT EXISTS idx_method_performance_recent ON generation_method_performance(method_name, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_method_performance_quality ON generation_method_performance(method_name, quality_score DESC);

-- Sample lifecycle queries
CREATE INDEX IF NOT EXISTS idx_synthetic_samples_lifecycle ON synthetic_data_samples(status, created_at);
CREATE INDEX IF NOT EXISTS idx_synthetic_samples_quality_domain ON synthetic_data_samples(domain_category, quality_score DESC);

-- ===================================
-- Views for Common Queries
-- ===================================

-- Recent generation performance view
CREATE OR REPLACE VIEW recent_generation_performance AS
SELECT 
    gs.session_id,
    gs.generation_method,
    gs.samples_generated,
    gs.average_quality_score,
    gs.generation_efficiency,
    gs.total_duration_seconds,
    gs.started_at,
    COUNT(gb.id) as batch_count,
    AVG(gb.efficiency_score) as avg_batch_efficiency,
    AVG(gb.throughput_samples_per_sec) as avg_throughput
FROM generation_sessions gs
LEFT JOIN generation_batches gb ON gs.session_id = gb.session_id
WHERE gs.started_at >= NOW() - INTERVAL '7 days'
GROUP BY gs.session_id, gs.generation_method, gs.samples_generated, 
         gs.average_quality_score, gs.generation_efficiency, 
         gs.total_duration_seconds, gs.started_at
ORDER BY gs.started_at DESC;

-- Method performance summary view
CREATE OR REPLACE VIEW method_performance_summary AS
SELECT 
    method_name,
    COUNT(*) as total_executions,
    AVG(quality_score) as avg_quality,
    AVG(diversity_score) as avg_diversity,
    AVG(success_rate) as avg_success_rate,
    AVG(generation_time_seconds) as avg_generation_time,
    SUM(samples_generated) as total_samples,
    MAX(recorded_at) as last_execution
FROM generation_method_performance
WHERE recorded_at >= NOW() - INTERVAL '30 days'
GROUP BY method_name
ORDER BY avg_quality DESC, avg_success_rate DESC;

-- Quality trends view
CREATE OR REPLACE VIEW quality_trends AS
SELECT 
    DATE_TRUNC('day', created_at) as trend_date,
    COUNT(*) as sessions_count,
    AVG(average_quality_score) as avg_quality,
    AVG(generation_efficiency) as avg_efficiency,
    SUM(samples_generated) as total_samples,
    AVG(total_duration_seconds) as avg_duration
FROM generation_sessions
WHERE created_at >= NOW() - INTERVAL '30 days'
  AND status = 'completed'
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY trend_date DESC;

-- ===================================
-- Triggers for Automatic Updates
-- ===================================

-- Update generation_sessions.updated_at on changes
CREATE OR REPLACE FUNCTION update_generation_sessions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_generation_sessions_updated_at_trigger
    BEFORE UPDATE ON generation_sessions
    FOR EACH ROW EXECUTE FUNCTION update_generation_sessions_updated_at();

-- Auto-update session summary when batches complete
CREATE OR REPLACE FUNCTION update_session_summary_on_batch_complete()
RETURNS TRIGGER AS $$
BEGIN
    -- Update session totals when a batch is inserted/updated
    UPDATE generation_sessions SET
        samples_generated = (
            SELECT COALESCE(SUM(samples_generated), 0)
            FROM generation_batches 
            WHERE session_id = NEW.session_id
        ),
        average_quality_score = (
            SELECT AVG(average_quality_score)
            FROM generation_batches 
            WHERE session_id = NEW.session_id 
              AND average_quality_score IS NOT NULL
        ),
        generation_efficiency = (
            SELECT AVG(efficiency_score)
            FROM generation_batches 
            WHERE session_id = NEW.session_id 
              AND efficiency_score IS NOT NULL
        ),
        updated_at = NOW()
    WHERE session_id = NEW.session_id;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_session_summary_trigger
    AFTER INSERT OR UPDATE ON generation_batches
    FOR EACH ROW EXECUTE FUNCTION update_session_summary_on_batch_complete();
