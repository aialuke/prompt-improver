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
-- Name: ab_experiments id; Type: DEFAULT; Schema: public; Owner: apes_user
--

ALTER TABLE ONLY public.ab_experiments ALTER COLUMN id SET DEFAULT nextval('public.ab_experiments_id_seq'::regclass);


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
-- Data for Name: discovered_patterns; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.discovered_patterns (id, pattern_id, pattern_name, pattern_description, pattern_rule, discovery_method, effectiveness_score, support_count, confidence_interval, validation_status, discovered_at, validated_at) FROM stdin;
\.


--
-- Data for Name: improvement_sessions; Type: TABLE DATA; Schema: public; Owner: apes_user
--

COPY public.improvement_sessions (id, session_id, user_id, original_prompt, final_prompt, rules_applied, iteration_count, total_improvement_score, session_metadata, started_at, completed_at, status) FROM stdin;
1	test_session_enhanced	\N	[Security audit only]	[Security audit only]	[]	1	\N	{"security_audit": {"redactions": 2, "security_level": "redacted", "audit_timestamp": "2025-07-13T13:29:46.392617", "redaction_details": {"email_address": {"count": 1, "placeholder": "[REDACTED_EMAIL_ADDRESS]"}, "openai_api_key": {"count": 1, "placeholder": "[REDACTED_OPENAI_API_KEY]"}}}}	2025-07-07 15:47:47.249983	\N	completed
2	174a3a81-86f2-4791-8e0b-6a464cbf542c	\N	Make this better	Make this better	[]	1	\N	null	2025-07-13 12:45:56.839329	\N	completed
3	807ad9e1-b8c3-4551-a0e8-e5ca18d775f6	\N	Write something good about AI	Write something good about AI	[]	1	\N	null	2025-07-13 12:46:43.257231	\N	completed
4	cf1eccb0-90de-4e45-a977-a218230ad978	\N	Analyze the impact of machine learning on society	Analyze the impact of machine learning on society	[]	1	\N	null	2025-07-13 12:46:43.286201	\N	completed
5	88f1da07-eeed-4e68-87b7-465f39dfc225	\N	How to optimize database queries?	How to optimize database queries?	[]	1	\N	null	2025-07-13 12:46:43.309042	\N	completed
6	b05278b0-8e78-43f7-8311-81373e1459ff	\N	Help me make better decisions for my company	Help me make better decisions for my company	[]	1	\N	null	2025-07-13 12:46:43.331757	\N	completed
7	667a45c5-07fb-4c18-9d96-0d6d79b6efe6	\N	Write something good about AI	Write something good about AI	[]	1	\N	null	2025-07-13 12:46:43.357377	\N	completed
8	2d97fa80-e9e5-44a9-9dac-b7a6ef06249f	\N	Analyze the impact of machine learning on society	Analyze the impact of machine learning on society	[]	1	\N	null	2025-07-13 12:46:43.377504	\N	completed
9	d79e2ac2-a851-4222-b538-c22e0e09d8c1	\N	How to optimize database queries?	How to optimize database queries?	[]	1	\N	null	2025-07-13 12:46:43.397654	\N	completed
10	6aa05126-8d6a-4b0d-bfad-0ec5be60c61f	\N	Help me make better decisions for my company	Help me make better decisions for my company	[]	1	\N	null	2025-07-13 12:46:43.417071	\N	completed
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
-- Name: ab_experiments_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.ab_experiments_id_seq', 1, false);


--
-- Name: discovered_patterns_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.discovered_patterns_id_seq', 1, false);


--
-- Name: improvement_sessions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: apes_user
--

SELECT pg_catalog.setval('public.improvement_sessions_id_seq', 10, true);


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
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: pg_database_owner
--

GRANT ALL ON SCHEMA public TO apes_app_role;


--
-- PostgreSQL database dump complete
--

