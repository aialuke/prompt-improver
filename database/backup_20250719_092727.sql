--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13 (Debian 15.13-1.pgdg120+1)
-- Dumped by pg_dump version 15.13 (Debian 15.13-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: btree_gin; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS btree_gin WITH SCHEMA public;


--
-- Name: EXTENSION btree_gin; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION btree_gin IS 'support for indexing common datatypes in GIN';


--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


--
-- Name: studydirection; Type: TYPE; Schema: public; Owner: apes_user
--

CREATE TYPE public.studydirection AS ENUM (
    'NOT_SET',
    'MINIMIZE',
    'MAXIMIZE'
);


ALTER TYPE public.studydirection OWNER TO apes_user;

--
-- Name: trialintermediatevaluetype; Type: TYPE; Schema: public; Owner: apes_user
--

CREATE TYPE public.trialintermediatevaluetype AS ENUM (
    'FINITE',
    'INF_POS',
    'INF_NEG',
    'NAN'
);


ALTER TYPE public.trialintermediatevaluetype OWNER TO apes_user;

--
-- Name: trialstate; Type: TYPE; Schema: public; Owner: apes_user
--

CREATE TYPE public.trialstate AS ENUM (
    'RUNNING',
    'COMPLETE',
    'PRUNED',
    'FAIL',
    'WAITING'
);


ALTER TYPE public.trialstate OWNER TO apes_user;

--
-- Name: trialvaluetype; Type: TYPE; Schema: public; Owner: apes_user
--

CREATE TYPE public.trialvaluetype AS ENUM (
    'FINITE',
    'INF_POS',
    'INF_NEG'
);


ALTER TYPE public.trialvaluetype OWNER TO apes_user;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: apes_user
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO apes_user;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: ab_experiments; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.ab_experiments (
    id integer NOT NULL,
    experiment_id uuid DEFAULT public.uuid_generate_v4(),
    experiment_name character varying(100) NOT NULL,
    description text,
    control_rules jsonb NOT NULL,
    treatment_rules jsonb NOT NULL,
    target_metric character varying(50),
    sample_size_per_group integer,
    current_sample_size integer DEFAULT 0,
    significance_threshold double precision DEFAULT 0.05,
    status character varying(20) DEFAULT 'running'::character varying,
    results jsonb,
    started_at timestamp without time zone DEFAULT now(),
    completed_at timestamp without time zone,
    CONSTRAINT ab_experiments_status_check CHECK (((status)::text = ANY ((ARRAY['planning'::character varying, 'running'::character varying, 'completed'::character varying, 'stopped'::character varying])::text[])))
);


ALTER TABLE public.ab_experiments OWNER TO apes_user;

--
-- Name: ab_experiments_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.ab_experiments_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ab_experiments_id_seq OWNER TO apes_user;

--
-- Name: ab_experiments_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.ab_experiments_id_seq OWNED BY public.ab_experiments.id;


--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO apes_user;

--
-- Name: auth_audit_log; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.auth_audit_log (
    id integer NOT NULL,
    user_id text,
    action text,
    resource_id text,
    details jsonb,
    "timestamp" timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.auth_audit_log OWNER TO apes_user;

--
-- Name: auth_audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.auth_audit_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.auth_audit_log_id_seq OWNER TO apes_user;

--
-- Name: auth_audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.auth_audit_log_id_seq OWNED BY public.auth_audit_log.id;


--
-- Name: auth_resources; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.auth_resources (
    resource_id text NOT NULL,
    owner_id text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.auth_resources OWNER TO apes_user;

--
-- Name: auth_roles; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.auth_roles (
    role_name text NOT NULL,
    description text
);


ALTER TABLE public.auth_roles OWNER TO apes_user;

--
-- Name: auth_user_roles; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.auth_user_roles (
    user_id text NOT NULL,
    role_name text NOT NULL,
    assigned_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.auth_user_roles OWNER TO apes_user;

--
-- Name: auth_users; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.auth_users (
    user_id text NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


ALTER TABLE public.auth_users OWNER TO apes_user;

--
-- Name: discovered_patterns; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.discovered_patterns (
    id integer NOT NULL,
    pattern_id uuid DEFAULT public.uuid_generate_v4(),
    pattern_name character varying(100),
    pattern_description text,
    pattern_rule jsonb NOT NULL,
    discovery_method character varying(50),
    effectiveness_score double precision,
    support_count integer,
    confidence_interval jsonb,
    validation_status character varying(20) DEFAULT 'pending'::character varying,
    discovered_at timestamp without time zone DEFAULT now(),
    validated_at timestamp without time zone,
    CONSTRAINT discovered_patterns_validation_status_check CHECK (((validation_status)::text = ANY ((ARRAY['pending'::character varying, 'validated'::character varying, 'rejected'::character varying])::text[])))
);


ALTER TABLE public.discovered_patterns OWNER TO apes_user;

--
-- Name: discovered_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.discovered_patterns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.discovered_patterns_id_seq OWNER TO apes_user;

--
-- Name: discovered_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.discovered_patterns_id_seq OWNED BY public.discovered_patterns.id;


--
-- Name: improvement_sessions; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.improvement_sessions (
    id integer NOT NULL,
    session_id character varying(100) NOT NULL,
    user_id character varying(50),
    original_prompt text NOT NULL,
    final_prompt text,
    rules_applied jsonb,
    iteration_count integer DEFAULT 1,
    total_improvement_score double precision,
    session_metadata jsonb,
    started_at timestamp without time zone DEFAULT now(),
    completed_at timestamp without time zone,
    status character varying(20) DEFAULT 'active'::character varying,
    CONSTRAINT improvement_sessions_status_check CHECK (((status)::text = ANY ((ARRAY['active'::character varying, 'completed'::character varying, 'abandoned'::character varying])::text[])))
);


ALTER TABLE public.improvement_sessions OWNER TO apes_user;

--
-- Name: improvement_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.improvement_sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.improvement_sessions_id_seq OWNER TO apes_user;

--
-- Name: improvement_sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.improvement_sessions_id_seq OWNED BY public.improvement_sessions.id;


--
-- Name: ml_model_performance; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.ml_model_performance (
    id integer NOT NULL,
    model_version character varying(50) NOT NULL,
    model_type character varying(50) NOT NULL,
    accuracy_score double precision,
    precision_score double precision,
    recall_score double precision,
    f1_score double precision,
    training_data_size integer,
    validation_data_size integer,
    hyperparameters jsonb,
    feature_importance jsonb,
    model_artifacts_path text,
    mlflow_run_id character varying(100),
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.ml_model_performance OWNER TO apes_user;

--
-- Name: ml_model_performance_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.ml_model_performance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ml_model_performance_id_seq OWNER TO apes_user;

--
-- Name: ml_model_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.ml_model_performance_id_seq OWNED BY public.ml_model_performance.id;


--
-- Name: rule_combinations; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.rule_combinations (
    id integer NOT NULL,
    combination_id uuid DEFAULT public.uuid_generate_v4(),
    rule_set jsonb NOT NULL,
    prompt_type character varying(50),
    combined_effectiveness double precision,
    individual_scores jsonb,
    sample_size integer DEFAULT 1,
    statistical_confidence double precision,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.rule_combinations OWNER TO apes_user;

--
-- Name: rule_combinations_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.rule_combinations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.rule_combinations_id_seq OWNER TO apes_user;

--
-- Name: rule_combinations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.rule_combinations_id_seq OWNED BY public.rule_combinations.id;


--
-- Name: rule_performance; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.rule_performance (
    id integer NOT NULL,
    rule_id character varying(50) NOT NULL,
    rule_name character varying(100) NOT NULL,
    prompt_id uuid DEFAULT public.uuid_generate_v4(),
    prompt_type character varying(50),
    prompt_category character varying(50),
    improvement_score double precision,
    confidence_level double precision,
    execution_time_ms integer,
    rule_parameters jsonb,
    prompt_characteristics jsonb,
    before_metrics jsonb,
    after_metrics jsonb,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    CONSTRAINT rule_performance_confidence_level_check CHECK (((confidence_level >= (0)::double precision) AND (confidence_level <= (1)::double precision))),
    CONSTRAINT rule_performance_improvement_score_check CHECK (((improvement_score >= (0)::double precision) AND (improvement_score <= (1)::double precision)))
);


ALTER TABLE public.rule_performance OWNER TO apes_user;

--
-- Name: rule_effectiveness_summary; Type: VIEW; Schema: public; Owner: apes_user
--

CREATE VIEW public.rule_effectiveness_summary AS
 SELECT rule_performance.rule_id,
    rule_performance.rule_name,
    count(*) AS usage_count,
    avg(rule_performance.improvement_score) AS avg_improvement,
    stddev(rule_performance.improvement_score) AS score_stddev,
    min(rule_performance.improvement_score) AS min_improvement,
    max(rule_performance.improvement_score) AS max_improvement,
    avg(rule_performance.confidence_level) AS avg_confidence,
    avg(rule_performance.execution_time_ms) AS avg_execution_time,
    count(DISTINCT rule_performance.prompt_type) AS prompt_types_count
   FROM public.rule_performance
  WHERE (rule_performance.created_at >= (now() - '30 days'::interval))
  GROUP BY rule_performance.rule_id, rule_performance.rule_name
  ORDER BY (avg(rule_performance.improvement_score)) DESC;


ALTER TABLE public.rule_effectiveness_summary OWNER TO apes_user;

--
-- Name: rule_metadata; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.rule_metadata (
    id integer NOT NULL,
    rule_id character varying(50) NOT NULL,
    rule_name character varying(100) NOT NULL,
    rule_category character varying(50),
    rule_description text,
    default_parameters jsonb,
    parameter_constraints jsonb,
    enabled boolean DEFAULT true,
    priority integer DEFAULT 100,
    rule_version character varying(20) DEFAULT '1.0.0'::character varying,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.rule_metadata OWNER TO apes_user;

--
-- Name: rule_metadata_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.rule_metadata_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.rule_metadata_id_seq OWNER TO apes_user;

--
-- Name: rule_metadata_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.rule_metadata_id_seq OWNED BY public.rule_metadata.id;


--
-- Name: rule_performance_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.rule_performance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.rule_performance_id_seq OWNER TO apes_user;

--
-- Name: rule_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.rule_performance_id_seq OWNED BY public.rule_performance.id;


--
-- Name: studies; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.studies (
    study_id integer NOT NULL,
    study_name character varying(512) NOT NULL
);


ALTER TABLE public.studies OWNER TO apes_user;

--
-- Name: studies_study_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.studies_study_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.studies_study_id_seq OWNER TO apes_user;

--
-- Name: studies_study_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.studies_study_id_seq OWNED BY public.studies.study_id;


--
-- Name: study_directions; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.study_directions (
    study_direction_id integer NOT NULL,
    direction public.studydirection NOT NULL,
    study_id integer NOT NULL,
    objective integer NOT NULL
);


ALTER TABLE public.study_directions OWNER TO apes_user;

--
-- Name: study_directions_study_direction_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.study_directions_study_direction_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.study_directions_study_direction_id_seq OWNER TO apes_user;

--
-- Name: study_directions_study_direction_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.study_directions_study_direction_id_seq OWNED BY public.study_directions.study_direction_id;


--
-- Name: study_system_attributes; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.study_system_attributes (
    study_system_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json text
);


ALTER TABLE public.study_system_attributes OWNER TO apes_user;

--
-- Name: study_system_attributes_study_system_attribute_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.study_system_attributes_study_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.study_system_attributes_study_system_attribute_id_seq OWNER TO apes_user;

--
-- Name: study_system_attributes_study_system_attribute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.study_system_attributes_study_system_attribute_id_seq OWNED BY public.study_system_attributes.study_system_attribute_id;


--
-- Name: study_user_attributes; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.study_user_attributes (
    study_user_attribute_id integer NOT NULL,
    study_id integer,
    key character varying(512),
    value_json text
);


ALTER TABLE public.study_user_attributes OWNER TO apes_user;

--
-- Name: study_user_attributes_study_user_attribute_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.study_user_attributes_study_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.study_user_attributes_study_user_attribute_id_seq OWNER TO apes_user;

--
-- Name: study_user_attributes_study_user_attribute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.study_user_attributes_study_user_attribute_id_seq OWNED BY public.study_user_attributes.study_user_attribute_id;


--
-- Name: training_prompts; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.training_prompts (
    id integer NOT NULL,
    prompt_text text NOT NULL,
    enhancement_result jsonb,
    data_source character varying(50) DEFAULT 'batch'::character varying,
    training_priority integer DEFAULT 50,
    created_at timestamp without time zone DEFAULT now(),
    session_id character varying(100)
);


ALTER TABLE public.training_prompts OWNER TO apes_user;

--
-- Name: training_prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.training_prompts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.training_prompts_id_seq OWNER TO apes_user;

--
-- Name: training_prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.training_prompts_id_seq OWNED BY public.training_prompts.id;


--
-- Name: trial_heartbeats; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_heartbeats (
    trial_heartbeat_id integer NOT NULL,
    trial_id integer NOT NULL,
    heartbeat timestamp without time zone NOT NULL
);


ALTER TABLE public.trial_heartbeats OWNER TO apes_user;

--
-- Name: trial_heartbeats_trial_heartbeat_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_heartbeats_trial_heartbeat_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_heartbeats_trial_heartbeat_id_seq OWNER TO apes_user;

--
-- Name: trial_heartbeats_trial_heartbeat_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_heartbeats_trial_heartbeat_id_seq OWNED BY public.trial_heartbeats.trial_heartbeat_id;


--
-- Name: trial_intermediate_values; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_intermediate_values (
    trial_intermediate_value_id integer NOT NULL,
    trial_id integer NOT NULL,
    step integer NOT NULL,
    intermediate_value double precision,
    intermediate_value_type public.trialintermediatevaluetype NOT NULL
);


ALTER TABLE public.trial_intermediate_values OWNER TO apes_user;

--
-- Name: trial_intermediate_values_trial_intermediate_value_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_intermediate_values_trial_intermediate_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_intermediate_values_trial_intermediate_value_id_seq OWNER TO apes_user;

--
-- Name: trial_intermediate_values_trial_intermediate_value_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_intermediate_values_trial_intermediate_value_id_seq OWNED BY public.trial_intermediate_values.trial_intermediate_value_id;


--
-- Name: trial_params; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_params (
    param_id integer NOT NULL,
    trial_id integer,
    param_name character varying(512),
    param_value double precision,
    distribution_json text
);


ALTER TABLE public.trial_params OWNER TO apes_user;

--
-- Name: trial_params_param_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_params_param_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_params_param_id_seq OWNER TO apes_user;

--
-- Name: trial_params_param_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_params_param_id_seq OWNED BY public.trial_params.param_id;


--
-- Name: trial_system_attributes; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_system_attributes (
    trial_system_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json text
);


ALTER TABLE public.trial_system_attributes OWNER TO apes_user;

--
-- Name: trial_system_attributes_trial_system_attribute_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_system_attributes_trial_system_attribute_id_seq OWNER TO apes_user;

--
-- Name: trial_system_attributes_trial_system_attribute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_system_attributes_trial_system_attribute_id_seq OWNED BY public.trial_system_attributes.trial_system_attribute_id;


--
-- Name: trial_user_attributes; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_user_attributes (
    trial_user_attribute_id integer NOT NULL,
    trial_id integer,
    key character varying(512),
    value_json text
);


ALTER TABLE public.trial_user_attributes OWNER TO apes_user;

--
-- Name: trial_user_attributes_trial_user_attribute_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_user_attributes_trial_user_attribute_id_seq OWNER TO apes_user;

--
-- Name: trial_user_attributes_trial_user_attribute_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_user_attributes_trial_user_attribute_id_seq OWNED BY public.trial_user_attributes.trial_user_attribute_id;


--
-- Name: trial_values; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trial_values (
    trial_value_id integer NOT NULL,
    trial_id integer NOT NULL,
    objective integer NOT NULL,
    value double precision,
    value_type public.trialvaluetype NOT NULL
);


ALTER TABLE public.trial_values OWNER TO apes_user;

--
-- Name: trial_values_trial_value_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trial_values_trial_value_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trial_values_trial_value_id_seq OWNER TO apes_user;

--
-- Name: trial_values_trial_value_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trial_values_trial_value_id_seq OWNED BY public.trial_values.trial_value_id;


--
-- Name: trials; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.trials (
    trial_id integer NOT NULL,
    number integer,
    study_id integer,
    state public.trialstate NOT NULL,
    datetime_start timestamp without time zone,
    datetime_complete timestamp without time zone
);


ALTER TABLE public.trials OWNER TO apes_user;

--
-- Name: trials_trial_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.trials_trial_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.trials_trial_id_seq OWNER TO apes_user;

--
-- Name: trials_trial_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.trials_trial_id_seq OWNED BY public.trials.trial_id;


--
-- Name: user_feedback; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.user_feedback (
    id integer NOT NULL,
    feedback_id uuid DEFAULT public.uuid_generate_v4(),
    original_prompt text NOT NULL,
    improved_prompt text NOT NULL,
    user_rating integer,
    applied_rules jsonb NOT NULL,
    user_context jsonb,
    improvement_areas jsonb,
    user_notes text,
    session_id character varying(100),
    created_at timestamp without time zone DEFAULT now(),
    CONSTRAINT user_feedback_user_rating_check CHECK (((user_rating >= 1) AND (user_rating <= 5)))
);


ALTER TABLE public.user_feedback OWNER TO apes_user;

--
-- Name: user_feedback_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.user_feedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_feedback_id_seq OWNER TO apes_user;

--
-- Name: user_feedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.user_feedback_id_seq OWNED BY public.user_feedback.id;


--
-- Name: user_satisfaction_summary; Type: VIEW; Schema: public; Owner: apes_user
--

CREATE VIEW public.user_satisfaction_summary AS
 SELECT date_trunc('day'::text, user_feedback.created_at) AS feedback_date,
    count(*) AS total_feedback,
    avg((user_feedback.user_rating)::double precision) AS avg_rating,
    count(
        CASE
            WHEN (user_feedback.user_rating >= 4) THEN 1
            ELSE NULL::integer
        END) AS positive_feedback,
    count(
        CASE
            WHEN (user_feedback.user_rating <= 2) THEN 1
            ELSE NULL::integer
        END) AS negative_feedback,
    array_agg(DISTINCT (user_feedback.applied_rules ->> 'rule_id'::text)) AS rules_used
   FROM public.user_feedback
  WHERE (user_feedback.created_at >= (now() - '90 days'::interval))
  GROUP BY (date_trunc('day'::text, user_feedback.created_at))
  ORDER BY (date_trunc('day'::text, user_feedback.created_at)) DESC;


ALTER TABLE public.user_satisfaction_summary OWNER TO apes_user;

--
-- Name: userfeedback; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.userfeedback (
    id integer NOT NULL,
    feedback_id uuid NOT NULL,
    original_prompt text NOT NULL,
    improved_prompt text NOT NULL,
    user_rating integer NOT NULL,
    applied_rules json NOT NULL,
    user_context json,
    improvement_areas json,
    user_notes text,
    session_id character varying(100),
    ml_optimized boolean DEFAULT false NOT NULL,
    model_id character varying(100),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT chk_user_rating_range CHECK (((user_rating >= 1) AND (user_rating <= 5)))
);


ALTER TABLE public.userfeedback OWNER TO apes_user;

--
-- Name: userfeedback_id_seq; Type: SEQUENCE; Schema: public; Owner: apes_user
--

CREATE SEQUENCE public.userfeedback_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.userfeedback_id_seq OWNER TO apes_user;

--
-- Name: userfeedback_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: apes_user
--

ALTER SEQUENCE public.userfeedback_id_seq OWNED BY public.userfeedback.id;


--
-- Name: version_info; Type: TABLE; Schema: public; Owner: apes_user
--

CREATE TABLE public.version_info (
    version_info_id integer NOT NULL,
    schema_version integer,
    library_version character varying(256),
    CONSTRAINT version_info_version_info_id_check CHECK ((version_info_id = 1))
);


ALTER TABLE public.version_info OWNER TO apes_user;

--
-- Name: ab_experiments id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.ab_experiments ALTER COLUMN id SET DEFAULT nextval('public.ab_experiments_id_seq'::regclass);


--
-- Name: auth_audit_log id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_audit_log ALTER COLUMN id SET DEFAULT nextval('public.auth_audit_log_id_seq'::regclass);


--
-- Name: discovered_patterns id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.discovered_patterns ALTER COLUMN id SET DEFAULT nextval('public.discovered_patterns_id_seq'::regclass);


--
-- Name: improvement_sessions id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.improvement_sessions ALTER COLUMN id SET DEFAULT nextval('public.improvement_sessions_id_seq'::regclass);


--
-- Name: ml_model_performance id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.ml_model_performance ALTER COLUMN id SET DEFAULT nextval('public.ml_model_performance_id_seq'::regclass);


--
-- Name: rule_combinations id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_combinations ALTER COLUMN id SET DEFAULT nextval('public.rule_combinations_id_seq'::regclass);


--
-- Name: rule_metadata id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_metadata ALTER COLUMN id SET DEFAULT nextval('public.rule_metadata_id_seq'::regclass);


--
-- Name: rule_performance id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_performance ALTER COLUMN id SET DEFAULT nextval('public.rule_performance_id_seq'::regclass);


--
-- Name: studies study_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.studies ALTER COLUMN study_id SET DEFAULT nextval('public.studies_study_id_seq'::regclass);


--
-- Name: study_directions study_direction_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_directions ALTER COLUMN study_direction_id SET DEFAULT nextval('public.study_directions_study_direction_id_seq'::regclass);


--
-- Name: study_system_attributes study_system_attribute_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_system_attributes ALTER COLUMN study_system_attribute_id SET DEFAULT nextval('public.study_system_attributes_study_system_attribute_id_seq'::regclass);


--
-- Name: study_user_attributes study_user_attribute_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_user_attributes ALTER COLUMN study_user_attribute_id SET DEFAULT nextval('public.study_user_attributes_study_user_attribute_id_seq'::regclass);


--
-- Name: training_prompts id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.training_prompts ALTER COLUMN id SET DEFAULT nextval('public.training_prompts_id_seq'::regclass);


--
-- Name: trial_heartbeats trial_heartbeat_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_heartbeats ALTER COLUMN trial_heartbeat_id SET DEFAULT nextval('public.trial_heartbeats_trial_heartbeat_id_seq'::regclass);


--
-- Name: trial_intermediate_values trial_intermediate_value_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_intermediate_values ALTER COLUMN trial_intermediate_value_id SET DEFAULT nextval('public.trial_intermediate_values_trial_intermediate_value_id_seq'::regclass);


--
-- Name: trial_params param_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_params ALTER COLUMN param_id SET DEFAULT nextval('public.trial_params_param_id_seq'::regclass);


--
-- Name: trial_system_attributes trial_system_attribute_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_system_attributes ALTER COLUMN trial_system_attribute_id SET DEFAULT nextval('public.trial_system_attributes_trial_system_attribute_id_seq'::regclass);


--
-- Name: trial_user_attributes trial_user_attribute_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_user_attributes ALTER COLUMN trial_user_attribute_id SET DEFAULT nextval('public.trial_user_attributes_trial_user_attribute_id_seq'::regclass);


--
-- Name: trial_values trial_value_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_values ALTER COLUMN trial_value_id SET DEFAULT nextval('public.trial_values_trial_value_id_seq'::regclass);


--
-- Name: trials trial_id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trials ALTER COLUMN trial_id SET DEFAULT nextval('public.trials_trial_id_seq'::regclass);


--
-- Name: user_feedback id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.user_feedback ALTER COLUMN id SET DEFAULT nextval('public.user_feedback_id_seq'::regclass);


--
-- Name: userfeedback id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.userfeedback ALTER COLUMN id SET DEFAULT nextval('public.userfeedback_id_seq'::regclass);


--
-- Data for Name: ab_experiments; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.ab_experiments (id, experiment_id, experiment_name, description, control_rules, treatment_rules, target_metric, sample_size_per_group, current_sample_size, significance_threshold, status, results, started_at, completed_at) FROM stdin;
\.


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.alembic_version (version_num) FROM stdin;
29408fe0f0d5
\.


--
-- Data for Name: auth_audit_log; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.auth_audit_log (id, user_id, action, resource_id, details, "timestamp") FROM stdin;
\.


--
-- Data for Name: auth_resources; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.auth_resources (resource_id, owner_id, created_at) FROM stdin;
\.


--
-- Data for Name: auth_roles; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.auth_roles (role_name, description) FROM stdin;
guest	Role: guest
user	Role: user
ml_analyst	Role: ml_analyst
ml_engineer	Role: ml_engineer
privacy_officer	Role: privacy_officer
security_analyst	Role: security_analyst
admin	Role: admin
super_admin	Role: super_admin
\.


--
-- Data for Name: auth_user_roles; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.auth_user_roles (user_id, role_name, assigned_at) FROM stdin;
\.


--
-- Data for Name: auth_users; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.auth_users (user_id, created_at) FROM stdin;
\.


--
-- Data for Name: discovered_patterns; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.discovered_patterns (id, pattern_id, pattern_name, pattern_description, pattern_rule, discovery_method, effectiveness_score, support_count, confidence_interval, validation_status, discovered_at, validated_at) FROM stdin;
\.


--
-- Data for Name: improvement_sessions; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.improvement_sessions (id, session_id, user_id, original_prompt, final_prompt, rules_applied, iteration_count, total_improvement_score, session_metadata, started_at, completed_at, status) FROM stdin;
15	audit_test_3	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.386727", "redaction_details": {"credit_card": {"count": 1, "placeholder": "[REDACTED_CREDIT_CARD]"}}}}	2025-07-15 03:04:02.901266	\N	completed
16	compliance_test_1	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.397216", "redaction_details": {"email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}}}}	2025-07-15 03:04:02.909918	\N	completed
2	174a3a81-86f2-4791-8e0b-6a464cbf542c	\N	Make this better	Make this better	[]	1	\N	null	2025-07-13 12:45:56.839329	\N	completed
3	807ad9e1-b8c3-4551-a0e8-e5ca18d775f6	\N	Write something good about AI	Write something good about AI	[]	1	\N	null	2025-07-13 12:46:43.257231	\N	completed
4	cf1eccb0-90de-4e45-a977-a218230ad978	\N	Analyze the impact of machine learning on society	Analyze the impact of machine learning on society	[]	1	\N	null	2025-07-13 12:46:43.286201	\N	completed
5	88f1da07-eeed-4e68-87b7-465f39dfc225	\N	How to optimize database queries?	How to optimize database queries?	[]	1	\N	null	2025-07-13 12:46:43.309042	\N	completed
6	b05278b0-8e78-43f7-8311-81373e1459ff	\N	Help me make better decisions for my company	Help me make better decisions for my company	[]	1	\N	null	2025-07-13 12:46:43.331757	\N	completed
7	667a45c5-07fb-4c18-9d96-0d6d79b6efe6	\N	Write something good about AI	Write something good about AI	[]	1	\N	null	2025-07-13 12:46:43.357377	\N	completed
8	2d97fa80-e9e5-44a9-9dac-b7a6ef06249f	\N	Analyze the impact of machine learning on society	Analyze the impact of machine learning on society	[]	1	\N	null	2025-07-13 12:46:43.377504	\N	completed
9	d79e2ac2-a851-4222-b538-c22e0e09d8c1	\N	How to optimize database queries?	How to optimize database queries?	[]	1	\N	null	2025-07-13 12:46:43.397654	\N	completed
10	6aa05126-8d6a-4b0d-bfad-0ec5be60c61f	\N	Help me make better decisions for my company	Help me make better decisions for my company	[]	1	\N	null	2025-07-13 12:46:43.417071	\N	completed
12	test_session	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 2, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.103722", "redaction_details": {"email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}, "openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-15 03:04:02.621279	\N	completed
17	compliance_test_2	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.404781", "redaction_details": {"openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-15 03:04:02.918162	\N	completed
11	integration_test_session	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.150786", "redaction_details": {"openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-15 03:04:02.554947	\N	completed
13	audit_test_1	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.362006", "redaction_details": {"email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}}}}	2025-07-15 03:04:02.882802	\N	completed
14	audit_test_2	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 1, "security_level": "redacted", "audit_timestamp": "2025-07-15T03:27:37.372953", "redaction_details": {"openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-15 03:04:02.892548	\N	completed
1	test_session_enhanced	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 2, "security_level": "redacted", "audit_timestamp": "2025-07-15T10:39:03.223293", "redaction_details": {"email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}, "openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-07 15:47:47.249983	\N	completed
\.


--
-- Data for Name: ml_model_performance; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.ml_model_performance (id, model_version, model_type, accuracy_score, precision_score, recall_score, f1_score, training_data_size, validation_data_size, hyperparameters, feature_importance, model_artifacts_path, mlflow_run_id, created_at) FROM stdin;
\.


--
-- Data for Name: rule_combinations; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.rule_combinations (id, combination_id, rule_set, prompt_type, combined_effectiveness, individual_scores, sample_size, statistical_confidence, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: rule_metadata; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.rule_metadata (id, rule_id, rule_name, rule_category, rule_description, default_parameters, parameter_constraints, enabled, priority, rule_version, created_at, updated_at) FROM stdin;
5	clarity_enhancement	Clarity Enhancement Rule	fundamental	Improves prompt clarity using research-validated patterns from Anthropic and OpenAI documentation. Replaces vague requests with specific, measurable outcomes and applies XML structure patterns.	{"min_clarity_score": 0.7, "use_structured_xml": true, "add_success_criteria": true, "apply_specificity_patterns": true, "context_placement_priority": "before_examples", "sentence_complexity_threshold": 20}	{"min_clarity_score": {"max": 1.0, "min": 0.0}, "use_structured_xml": {"type": "boolean"}, "apply_specificity_patterns": {"type": "boolean"}, "sentence_complexity_threshold": {"max": 50, "min": 10}}	t	10	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
6	specificity_enhancement	Specificity and Detail Rule	fundamental	Reduces vague language and increases prompt specificity using multi-source research patterns. Enforces measurable goals and specific success criteria.	{"avoid_hedge_words": true, "specificity_patterns": ["who_what_when_where", "concrete_examples", "quantifiable_metrics"], "enforce_measurable_goals": true, "include_success_criteria": true, "vague_language_threshold": 0.3, "require_specific_outcomes": true}	{"enforce_measurable_goals": {"type": "boolean"}, "include_success_criteria": {"type": "boolean"}, "vague_language_threshold": {"max": 1.0, "min": 0.0}, "require_specific_outcomes": {"type": "boolean"}}	t	9	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
7	chain_of_thought	Chain of Thought Reasoning Rule	reasoning	Implements step-by-step reasoning patterns based on CoT research across multiple LLM providers. Uses zero-shot and few-shot CoT techniques with structured thinking tags.	{"use_thinking_tags": true, "zero_shot_trigger": "Let's think step by step", "enable_step_by_step": true, "min_reasoning_steps": 3, "use_structured_response": true, "encourage_explicit_reasoning": true}	{"use_thinking_tags": {"type": "boolean"}, "enable_step_by_step": {"type": "boolean"}, "min_reasoning_steps": {"max": 10, "min": 1}}	t	8	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
8	few_shot_examples	Few-Shot Example Integration Rule	examples	Incorporates 2-5 optimal examples based on research from PromptHub and OpenAI documentation. Uses diverse examples with XML delimiters and balanced positive/negative cases.	{"example_placement": "after_context", "use_xml_delimiters": true, "optimal_example_count": 3, "require_diverse_examples": true, "include_negative_examples": true, "recency_bias_optimization": true}	{"optimal_example_count": {"max": 5, "min": 2}, "require_diverse_examples": {"type": "boolean"}, "include_negative_examples": {"type": "boolean"}}	t	7	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
9	role_based_prompting	Expert Role Assignment Rule	context	Assigns appropriate expert personas based on Anthropic best practices for role-based prompting. Automatically detects domain and maintains persona consistency.	{"expertise_depth": "senior_level", "auto_detect_domain": true, "use_system_prompts": true, "include_credentials": true, "maintain_persona_consistency": true}	{"expertise_depth": {"values": ["junior", "mid_level", "senior_level", "expert"]}, "auto_detect_domain": {"type": "boolean"}, "use_system_prompts": {"type": "boolean"}}	t	6	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
10	xml_structure_enhancement	XML Structure Enhancement Rule	structure	Implements XML tagging patterns recommended by Anthropic for Claude optimization. Uses context, instruction, example, thinking, and response tags for better organization.	{"attribute_usage": "minimal", "use_context_tags": true, "use_example_tags": true, "use_response_tags": true, "use_thinking_tags": true, "use_instruction_tags": true, "nested_structure_allowed": true}	{"use_context_tags": {"type": "boolean"}, "use_example_tags": {"type": "boolean"}, "use_response_tags": {"type": "boolean"}, "use_thinking_tags": {"type": "boolean"}, "use_instruction_tags": {"type": "boolean"}}	t	5	1.0.0	2025-07-13 12:45:45.212906	2025-07-13 12:45:45.212906
\.


--
-- Data for Name: rule_performance; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.rule_performance (id, rule_id, rule_name, prompt_id, prompt_type, prompt_category, improvement_score, confidence_level, execution_time_ms, rule_parameters, prompt_characteristics, before_metrics, after_metrics, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: studies; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.studies (study_id, study_name) FROM stdin;
\.


--
-- Data for Name: study_directions; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.study_directions (study_direction_id, direction, study_id, objective) FROM stdin;
\.


--
-- Data for Name: study_system_attributes; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.study_system_attributes (study_system_attribute_id, study_id, key, value_json) FROM stdin;
\.


--
-- Data for Name: study_user_attributes; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.study_user_attributes (study_user_attribute_id, study_id, key, value_json) FROM stdin;
\.


--
-- Data for Name: training_prompts; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.training_prompts (id, prompt_text, enhancement_result, data_source, training_priority, created_at, session_id) FROM stdin;
\.


--
-- Data for Name: trial_heartbeats; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_heartbeats (trial_heartbeat_id, trial_id, heartbeat) FROM stdin;
\.


--
-- Data for Name: trial_intermediate_values; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_intermediate_values (trial_intermediate_value_id, trial_id, step, intermediate_value, intermediate_value_type) FROM stdin;
\.


--
-- Data for Name: trial_params; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_params (param_id, trial_id, param_name, param_value, distribution_json) FROM stdin;
\.


--
-- Data for Name: trial_system_attributes; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_system_attributes (trial_system_attribute_id, trial_id, key, value_json) FROM stdin;
\.


--
-- Data for Name: trial_user_attributes; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_user_attributes (trial_user_attribute_id, trial_id, key, value_json) FROM stdin;
\.


--
-- Data for Name: trial_values; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trial_values (trial_value_id, trial_id, objective, value, value_type) FROM stdin;
\.


--
-- Data for Name: trials; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.trials (trial_id, number, study_id, state, datetime_start, datetime_complete) FROM stdin;
\.


--
-- Data for Name: user_feedback; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.user_feedback (id, feedback_id, original_prompt, improved_prompt, user_rating, applied_rules, user_context, improvement_areas, user_notes, session_id, created_at) FROM stdin;
\.


--
-- Data for Name: userfeedback; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.userfeedback (id, feedback_id, original_prompt, improved_prompt, user_rating, applied_rules, user_context, improvement_areas, user_notes, session_id, ml_optimized, model_id, created_at) FROM stdin;
\.


--
-- Data for Name: version_info; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.version_info (version_info_id, schema_version, library_version) FROM stdin;
1	12	4.4.0
\.


--
-- Name: ab_experiments_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.ab_experiments_id_seq', 1, false);


--
-- Name: auth_audit_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.auth_audit_log_id_seq', 1566, true);


--
-- Name: discovered_patterns_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.discovered_patterns_id_seq', 1, false);


--
-- Name: improvement_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.improvement_sessions_id_seq', 17, true);


--
-- Name: ml_model_performance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.ml_model_performance_id_seq', 1, false);


--
-- Name: rule_combinations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.rule_combinations_id_seq', 1, false);


--
-- Name: rule_metadata_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.rule_metadata_id_seq', 10, true);


--
-- Name: rule_performance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.rule_performance_id_seq', 1, false);


--
-- Name: studies_study_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.studies_study_id_seq', 1, false);


--
-- Name: study_directions_study_direction_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.study_directions_study_direction_id_seq', 1, false);


--
-- Name: study_system_attributes_study_system_attribute_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.study_system_attributes_study_system_attribute_id_seq', 1, false);


--
-- Name: study_user_attributes_study_user_attribute_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.study_user_attributes_study_user_attribute_id_seq', 1, false);


--
-- Name: training_prompts_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.training_prompts_id_seq', 1, false);


--
-- Name: trial_heartbeats_trial_heartbeat_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_heartbeats_trial_heartbeat_id_seq', 1, false);


--
-- Name: trial_intermediate_values_trial_intermediate_value_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_intermediate_values_trial_intermediate_value_id_seq', 1, false);


--
-- Name: trial_params_param_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_params_param_id_seq', 1, false);


--
-- Name: trial_system_attributes_trial_system_attribute_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_system_attributes_trial_system_attribute_id_seq', 1, false);


--
-- Name: trial_user_attributes_trial_user_attribute_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_user_attributes_trial_user_attribute_id_seq', 1, false);


--
-- Name: trial_values_trial_value_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trial_values_trial_value_id_seq', 1, false);


--
-- Name: trials_trial_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.trials_trial_id_seq', 1, false);


--
-- Name: user_feedback_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.user_feedback_id_seq', 1, false);


--
-- Name: userfeedback_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.userfeedback_id_seq', 1, false);


--
-- Name: ab_experiments ab_experiments_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.ab_experiments
    ADD CONSTRAINT ab_experiments_pkey PRIMARY KEY (id);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: auth_audit_log auth_audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_audit_log
    ADD CONSTRAINT auth_audit_log_pkey PRIMARY KEY (id);


--
-- Name: auth_resources auth_resources_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_resources
    ADD CONSTRAINT auth_resources_pkey PRIMARY KEY (resource_id);


--
-- Name: auth_roles auth_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_roles
    ADD CONSTRAINT auth_roles_pkey PRIMARY KEY (role_name);


--
-- Name: auth_user_roles auth_user_roles_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_user_roles
    ADD CONSTRAINT auth_user_roles_pkey PRIMARY KEY (user_id, role_name);


--
-- Name: auth_users auth_users_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_users
    ADD CONSTRAINT auth_users_pkey PRIMARY KEY (user_id);


--
-- Name: discovered_patterns discovered_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.discovered_patterns
    ADD CONSTRAINT discovered_patterns_pkey PRIMARY KEY (id);


--
-- Name: improvement_sessions improvement_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.improvement_sessions
    ADD CONSTRAINT improvement_sessions_pkey PRIMARY KEY (id);


--
-- Name: improvement_sessions improvement_sessions_session_id_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.improvement_sessions
    ADD CONSTRAINT improvement_sessions_session_id_key UNIQUE (session_id);


--
-- Name: ml_model_performance ml_model_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.ml_model_performance
    ADD CONSTRAINT ml_model_performance_pkey PRIMARY KEY (id);


--
-- Name: rule_combinations rule_combinations_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_combinations
    ADD CONSTRAINT rule_combinations_pkey PRIMARY KEY (id);


--
-- Name: rule_metadata rule_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_metadata
    ADD CONSTRAINT rule_metadata_pkey PRIMARY KEY (id);


--
-- Name: rule_metadata rule_metadata_rule_id_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_metadata
    ADD CONSTRAINT rule_metadata_rule_id_key UNIQUE (rule_id);


--
-- Name: rule_performance rule_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.rule_performance
    ADD CONSTRAINT rule_performance_pkey PRIMARY KEY (id);


--
-- Name: studies studies_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.studies
    ADD CONSTRAINT studies_pkey PRIMARY KEY (study_id);


--
-- Name: study_directions study_directions_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_pkey PRIMARY KEY (study_direction_id);


--
-- Name: study_directions study_directions_study_id_objective_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_study_id_objective_key UNIQUE (study_id, objective);


--
-- Name: study_system_attributes study_system_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_pkey PRIMARY KEY (study_system_attribute_id);


--
-- Name: study_system_attributes study_system_attributes_study_id_key_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_key_key UNIQUE (study_id, key);


--
-- Name: study_user_attributes study_user_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_pkey PRIMARY KEY (study_user_attribute_id);


--
-- Name: study_user_attributes study_user_attributes_study_id_key_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_key_key UNIQUE (study_id, key);


--
-- Name: training_prompts training_prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.training_prompts
    ADD CONSTRAINT training_prompts_pkey PRIMARY KEY (id);


--
-- Name: trial_heartbeats trial_heartbeats_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_heartbeats
    ADD CONSTRAINT trial_heartbeats_pkey PRIMARY KEY (trial_heartbeat_id);


--
-- Name: trial_heartbeats trial_heartbeats_trial_id_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_heartbeats
    ADD CONSTRAINT trial_heartbeats_trial_id_key UNIQUE (trial_id);


--
-- Name: trial_intermediate_values trial_intermediate_values_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_pkey PRIMARY KEY (trial_intermediate_value_id);


--
-- Name: trial_intermediate_values trial_intermediate_values_trial_id_step_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_trial_id_step_key UNIQUE (trial_id, step);


--
-- Name: trial_params trial_params_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_pkey PRIMARY KEY (param_id);


--
-- Name: trial_params trial_params_trial_id_param_name_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_param_name_key UNIQUE (trial_id, param_name);


--
-- Name: trial_system_attributes trial_system_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_pkey PRIMARY KEY (trial_system_attribute_id);


--
-- Name: trial_system_attributes trial_system_attributes_trial_id_key_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_key_key UNIQUE (trial_id, key);


--
-- Name: trial_user_attributes trial_user_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_pkey PRIMARY KEY (trial_user_attribute_id);


--
-- Name: trial_user_attributes trial_user_attributes_trial_id_key_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_key_key UNIQUE (trial_id, key);


--
-- Name: trial_values trial_values_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_pkey PRIMARY KEY (trial_value_id);


--
-- Name: trial_values trial_values_trial_id_objective_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_objective_key UNIQUE (trial_id, objective);


--
-- Name: trials trials_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_pkey PRIMARY KEY (trial_id);


--
-- Name: user_feedback user_feedback_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.user_feedback
    ADD CONSTRAINT user_feedback_pkey PRIMARY KEY (id);


--
-- Name: userfeedback userfeedback_feedback_id_key; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.userfeedback
    ADD CONSTRAINT userfeedback_feedback_id_key UNIQUE (feedback_id);


--
-- Name: userfeedback userfeedback_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.userfeedback
    ADD CONSTRAINT userfeedback_pkey PRIMARY KEY (id);


--
-- Name: version_info version_info_pkey; Type: CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.version_info
    ADD CONSTRAINT version_info_pkey PRIMARY KEY (version_info_id);


--
-- Name: idx_discovered_patterns_effectiveness; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_discovered_patterns_effectiveness ON public.discovered_patterns USING btree (effectiveness_score);


--
-- Name: idx_discovered_patterns_status; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_discovered_patterns_status ON public.discovered_patterns USING btree (validation_status);


--
-- Name: idx_improvement_sessions_started_at; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_improvement_sessions_started_at ON public.improvement_sessions USING btree (started_at);


--
-- Name: idx_improvement_sessions_status; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_improvement_sessions_status ON public.improvement_sessions USING btree (status);


--
-- Name: idx_improvement_sessions_user_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_improvement_sessions_user_id ON public.improvement_sessions USING btree (user_id);


--
-- Name: idx_ml_model_performance_type; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_ml_model_performance_type ON public.ml_model_performance USING btree (model_type);


--
-- Name: idx_ml_model_performance_version; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_ml_model_performance_version ON public.ml_model_performance USING btree (model_version);


--
-- Name: idx_rule_performance_characteristics; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_rule_performance_characteristics ON public.rule_performance USING gin (prompt_characteristics);


--
-- Name: idx_rule_performance_created_at; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_rule_performance_created_at ON public.rule_performance USING btree (created_at);


--
-- Name: idx_rule_performance_improvement_score; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_rule_performance_improvement_score ON public.rule_performance USING btree (improvement_score);


--
-- Name: idx_rule_performance_prompt_type; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_rule_performance_prompt_type ON public.rule_performance USING btree (prompt_type);


--
-- Name: idx_rule_performance_rule_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_rule_performance_rule_id ON public.rule_performance USING btree (rule_id);


--
-- Name: idx_training_prompts_created_at; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_training_prompts_created_at ON public.training_prompts USING btree (created_at);


--
-- Name: idx_training_prompts_data_source; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_training_prompts_data_source ON public.training_prompts USING btree (data_source);


--
-- Name: idx_training_prompts_priority; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_training_prompts_priority ON public.training_prompts USING btree (training_priority);


--
-- Name: idx_training_prompts_session_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_training_prompts_session_id ON public.training_prompts USING btree (session_id);


--
-- Name: idx_user_feedback_applied_rules; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_user_feedback_applied_rules ON public.user_feedback USING gin (applied_rules);


--
-- Name: idx_user_feedback_created_at; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_user_feedback_created_at ON public.user_feedback USING btree (created_at);


--
-- Name: idx_user_feedback_rating; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_user_feedback_rating ON public.user_feedback USING btree (user_rating);


--
-- Name: idx_user_feedback_session_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX idx_user_feedback_session_id ON public.user_feedback USING btree (session_id);


--
-- Name: ix_studies_study_name; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE UNIQUE INDEX ix_studies_study_name ON public.studies USING btree (study_name);


--
-- Name: ix_trials_study_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX ix_trials_study_id ON public.trials USING btree (study_id);


--
-- Name: ix_userfeedback_created_at; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX ix_userfeedback_created_at ON public.userfeedback USING btree (created_at);


--
-- Name: ix_userfeedback_session_id; Type: INDEX; Schema: public; Owner: apes_user
--

CREATE INDEX ix_userfeedback_session_id ON public.userfeedback USING btree (session_id);


--
-- Name: rule_combinations update_rule_combinations_updated_at; Type: TRIGGER; Schema: public; Owner: apes_user
--

CREATE TRIGGER update_rule_combinations_updated_at BEFORE UPDATE ON public.rule_combinations FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: rule_metadata update_rule_metadata_updated_at; Type: TRIGGER; Schema: public; Owner: apes_user
--

CREATE TRIGGER update_rule_metadata_updated_at BEFORE UPDATE ON public.rule_metadata FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: rule_performance update_rule_performance_updated_at; Type: TRIGGER; Schema: public; Owner: apes_user
--

CREATE TRIGGER update_rule_performance_updated_at BEFORE UPDATE ON public.rule_performance FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: auth_resources auth_resources_owner_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_resources
    ADD CONSTRAINT auth_resources_owner_id_fkey FOREIGN KEY (owner_id) REFERENCES public.auth_users(user_id);


--
-- Name: auth_user_roles auth_user_roles_role_name_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_user_roles
    ADD CONSTRAINT auth_user_roles_role_name_fkey FOREIGN KEY (role_name) REFERENCES public.auth_roles(role_name);


--
-- Name: auth_user_roles auth_user_roles_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.auth_user_roles
    ADD CONSTRAINT auth_user_roles_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.auth_users(user_id);


--
-- Name: study_directions study_directions_study_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_directions
    ADD CONSTRAINT study_directions_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);


--
-- Name: study_system_attributes study_system_attributes_study_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_system_attributes
    ADD CONSTRAINT study_system_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);


--
-- Name: study_user_attributes study_user_attributes_study_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.study_user_attributes
    ADD CONSTRAINT study_user_attributes_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);


--
-- Name: trial_heartbeats trial_heartbeats_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_heartbeats
    ADD CONSTRAINT trial_heartbeats_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trial_intermediate_values trial_intermediate_values_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_intermediate_values
    ADD CONSTRAINT trial_intermediate_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trial_params trial_params_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_params
    ADD CONSTRAINT trial_params_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trial_system_attributes trial_system_attributes_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_system_attributes
    ADD CONSTRAINT trial_system_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trial_user_attributes trial_user_attributes_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_user_attributes
    ADD CONSTRAINT trial_user_attributes_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trial_values trial_values_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trial_values
    ADD CONSTRAINT trial_values_trial_id_fkey FOREIGN KEY (trial_id) REFERENCES public.trials(trial_id);


--
-- Name: trials trials_study_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.trials
    ADD CONSTRAINT trials_study_id_fkey FOREIGN KEY (study_id) REFERENCES public.studies(study_id);


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO apes_app_role;


--
-- PostgreSQL database dump complete
--

