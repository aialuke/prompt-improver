-- APES Telemetry Schema for JSONB-Compatible OpenTelemetry Data
-- Integrates with existing PostgreSQL JSONB infrastructure
-- Created: 2025-01-30

-- ===================================
-- OpenTelemetry Telemetry Storage
-- ===================================

-- Store OpenTelemetry metrics with JSONB compatibility
CREATE TABLE otel_metrics (
    id SERIAL PRIMARY KEY,
    metric_id UUID DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(200) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'gauge', 'counter', 'histogram', 'summary'
    metric_value FLOAT NOT NULL,
    metric_labels JSONB NOT NULL DEFAULT '{}',
    metric_attributes JSONB NOT NULL DEFAULT '{}',
    resource_attributes JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store OpenTelemetry traces with JSONB compatibility
CREATE TABLE otel_traces (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(32) NOT NULL,
    span_id VARCHAR(16) NOT NULL,
    parent_span_id VARCHAR(16),
    service_name VARCHAR(100) NOT NULL,
    operation_name VARCHAR(200) NOT NULL,
    span_kind VARCHAR(50), -- 'server', 'client', 'producer', 'consumer', 'internal'
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    status_code VARCHAR(20), -- 'ok', 'error', 'timeout'
    status_message TEXT,
    span_attributes JSONB NOT NULL DEFAULT '{}',
    resource_attributes JSONB NOT NULL DEFAULT '{}',
    events JSONB NOT NULL DEFAULT '[]',
    links JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store OpenTelemetry logs with JSONB compatibility
CREATE TABLE otel_logs (
    id SERIAL PRIMARY KEY,
    log_id UUID DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    severity_text VARCHAR(20),
    severity_number INTEGER,
    log_body TEXT,
    log_attributes JSONB NOT NULL DEFAULT '{}',
    resource_attributes JSONB NOT NULL DEFAULT '{}',
    trace_id VARCHAR(32),
    span_id VARCHAR(16),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    observed_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store CI/CD pipeline metrics with JSONB compatibility
CREATE TABLE cicd_metrics (
    id SERIAL PRIMARY KEY,
    pipeline_id VARCHAR(100) NOT NULL,
    pipeline_name VARCHAR(100) NOT NULL,
    build_number INTEGER,
    commit_sha VARCHAR(40),
    branch_name VARCHAR(100),
    pipeline_status VARCHAR(20), -- 'success', 'failure', 'running', 'cancelled'
    stage_name VARCHAR(100),
    job_name VARCHAR(100),
    metrics_data JSONB NOT NULL,
    alerts_data JSONB NOT NULL DEFAULT '[]',
    duration_seconds INTEGER,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store ML monitoring data with JSONB compatibility
CREATE TABLE ml_monitoring (
    id SERIAL PRIMARY KEY,
    monitoring_id UUID DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    monitoring_type VARCHAR(50) NOT NULL, -- 'drift', 'performance', 'inference', 'training'
    metrics_data JSONB NOT NULL,
    drift_analysis JSONB NOT NULL DEFAULT '{}',
    performance_metrics JSONB NOT NULL DEFAULT '{}',
    alert_conditions JSONB NOT NULL DEFAULT '[]',
    threshold_violations JSONB NOT NULL DEFAULT '[]',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Store health monitoring data with JSONB compatibility
CREATE TABLE health_monitoring (
    id SERIAL PRIMARY KEY,
    health_check_id UUID DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL, -- 'liveness', 'readiness', 'startup', 'deep'
    status VARCHAR(20) NOT NULL, -- 'healthy', 'unhealthy', 'degraded', 'unknown'
    response_time_ms INTEGER,
    check_details JSONB NOT NULL DEFAULT '{}',
    dependencies_status JSONB NOT NULL DEFAULT '{}',
    error_details JSONB NOT NULL DEFAULT '{}',
    sla_compliance BOOLEAN DEFAULT true,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===================================
-- Indexes for Performance
-- ===================================

-- OpenTelemetry metrics indexes
CREATE INDEX idx_otel_metrics_service_name ON otel_metrics(service_name);
CREATE INDEX idx_otel_metrics_metric_name ON otel_metrics(metric_name);
CREATE INDEX idx_otel_metrics_timestamp ON otel_metrics(timestamp);
CREATE INDEX idx_otel_metrics_labels ON otel_metrics USING GIN (metric_labels);
CREATE INDEX idx_otel_metrics_attributes ON otel_metrics USING GIN (metric_attributes);

-- OpenTelemetry traces indexes
CREATE INDEX idx_otel_traces_trace_id ON otel_traces(trace_id);
CREATE INDEX idx_otel_traces_span_id ON otel_traces(span_id);
CREATE INDEX idx_otel_traces_service_name ON otel_traces(service_name);
CREATE INDEX idx_otel_traces_operation_name ON otel_traces(operation_name);
CREATE INDEX idx_otel_traces_start_time ON otel_traces(start_time);
CREATE INDEX idx_otel_traces_duration ON otel_traces(duration_ms);
CREATE INDEX idx_otel_traces_attributes ON otel_traces USING GIN (span_attributes);

-- OpenTelemetry logs indexes
CREATE INDEX idx_otel_logs_service_name ON otel_logs(service_name);
CREATE INDEX idx_otel_logs_severity ON otel_logs(severity_number);
CREATE INDEX idx_otel_logs_timestamp ON otel_logs(timestamp);
CREATE INDEX idx_otel_logs_trace_id ON otel_logs(trace_id);
CREATE INDEX idx_otel_logs_attributes ON otel_logs USING GIN (log_attributes);

-- CI/CD metrics indexes
CREATE INDEX idx_cicd_metrics_pipeline_id ON cicd_metrics(pipeline_id);
CREATE INDEX idx_cicd_metrics_status ON cicd_metrics(pipeline_status);
CREATE INDEX idx_cicd_metrics_branch ON cicd_metrics(branch_name);
CREATE INDEX idx_cicd_metrics_started_at ON cicd_metrics(started_at);
CREATE INDEX idx_cicd_metrics_data ON cicd_metrics USING GIN (metrics_data);

-- ML monitoring indexes
CREATE INDEX idx_ml_monitoring_model_name ON ml_monitoring(model_name);
CREATE INDEX idx_ml_monitoring_type ON ml_monitoring(monitoring_type);
CREATE INDEX idx_ml_monitoring_timestamp ON ml_monitoring(timestamp);
CREATE INDEX idx_ml_monitoring_metrics ON ml_monitoring USING GIN (metrics_data);

-- Health monitoring indexes
CREATE INDEX idx_health_monitoring_service ON health_monitoring(service_name);
CREATE INDEX idx_health_monitoring_status ON health_monitoring(status);
CREATE INDEX idx_health_monitoring_timestamp ON health_monitoring(timestamp);
CREATE INDEX idx_health_monitoring_sla ON health_monitoring(sla_compliance);

-- ===================================
-- Views for Common Queries
-- ===================================

-- Service health summary view
CREATE VIEW service_health_summary AS
SELECT 
    service_name,
    COUNT(*) as total_checks,
    COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_checks,
    COUNT(CASE WHEN status = 'unhealthy' THEN 1 END) as unhealthy_checks,
    AVG(response_time_ms) as avg_response_time,
    MAX(timestamp) as last_check_time,
    (COUNT(CASE WHEN status = 'healthy' THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100) as health_percentage
FROM health_monitoring 
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY service_name;

-- ML model performance summary view
CREATE VIEW ml_model_summary AS
SELECT 
    model_name,
    model_version,
    monitoring_type,
    COUNT(*) as monitoring_events,
    AVG((metrics_data->>'accuracy')::FLOAT) as avg_accuracy,
    COUNT(CASE WHEN (threshold_violations::TEXT != '[]') THEN 1 END) as violations_count,
    MAX(timestamp) as last_monitoring_time
FROM ml_monitoring 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY model_name, model_version, monitoring_type;

-- CI/CD pipeline success rate view
CREATE VIEW cicd_success_rate AS
SELECT 
    pipeline_name,
    branch_name,
    COUNT(*) as total_runs,
    COUNT(CASE WHEN pipeline_status = 'success' THEN 1 END) as successful_runs,
    (COUNT(CASE WHEN pipeline_status = 'success' THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100) as success_rate,
    AVG(duration_seconds) as avg_duration_seconds,
    MAX(completed_at) as last_run_time
FROM cicd_metrics 
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY pipeline_name, branch_name;

-- ===================================
-- Functions for JSONB Operations
-- ===================================

-- Function to extract metric value by path
CREATE OR REPLACE FUNCTION extract_metric_value(
    metrics_jsonb JSONB,
    metric_path TEXT
) RETURNS FLOAT AS $$
BEGIN
    RETURN (metrics_jsonb #>> string_to_array(metric_path, '.'))::FLOAT;
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to check SLA compliance
CREATE OR REPLACE FUNCTION check_sla_compliance(
    response_time_ms INTEGER,
    sla_threshold_ms INTEGER DEFAULT 200
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN response_time_ms <= sla_threshold_ms;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate JSONB metrics
CREATE OR REPLACE FUNCTION aggregate_jsonb_metrics(
    metrics_array JSONB[]
) RETURNS JSONB AS $$
DECLARE
    result JSONB := '{}';
    metric JSONB;
    key TEXT;
    value NUMERIC;
    sum_value NUMERIC;
    count_value INTEGER;
BEGIN
    FOR metric IN SELECT unnest(metrics_array) LOOP
        FOR key IN SELECT jsonb_object_keys(metric) LOOP
            value := (metric ->> key)::NUMERIC;
            sum_value := COALESCE((result ->> (key || '_sum'))::NUMERIC, 0) + value;
            count_value := COALESCE((result ->> (key || '_count'))::INTEGER, 0) + 1;
            
            result := result || jsonb_build_object(
                key || '_sum', sum_value,
                key || '_count', count_value,
                key || '_avg', sum_value / count_value
            );
        END LOOP;
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
