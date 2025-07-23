"""
Component definitions for all 50+ ML components across 6 tiers.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field

from ..core.component_registry import ComponentTier, ComponentInfo, ComponentCapability


@dataclass
class ComponentDefinitions:
    """Central registry of all ML component definitions."""
    
    # Tier 1: Core ML Pipeline Components (11 components)
    tier1_core_components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "training_data_loader": {
            "description": "Central training data hub for ML pipeline",
            "file_path": "ml/core/training_data_loader.py",
            "capabilities": ["data_loading", "data_preprocessing", "training_data_management"],
            "dependencies": ["database", "file_system"],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "ml_integration": {
            "description": "Core ML service processing engine",
            "file_path": "ml/core/ml_integration.py", 
            "capabilities": ["model_training", "prediction", "inference"],
            "dependencies": ["training_data_loader", "model_manager"],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores", "gpu": "optional"}
        },
        "rule_optimizer": {
            "description": "Multi-objective optimization for rules",
            "file_path": "ml/optimization/algorithms/rule_optimizer.py",
            "capabilities": ["optimization", "rule_tuning", "parameter_search"],
            "dependencies": ["ml_integration"],
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "multi_armed_bandit": {
            "description": "Thompson Sampling and UCB algorithms",
            "file_path": "ml/optimization/algorithms/multi_armed_bandit.py",
            "capabilities": ["exploration", "exploitation", "adaptive_selection"],
            "dependencies": [],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "apriori_analyzer": {
            "description": "Association rule mining and pattern discovery",
            "file_path": "ml/learning/patterns/apriori_analyzer.py",
            "capabilities": ["pattern_mining", "association_rules", "frequent_patterns"],
            "dependencies": ["training_data_loader"],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "batch_processor": {
            "description": "Batch training processing coordinator",
            "file_path": "ml/optimization/batch/batch_processor.py",
            "capabilities": ["batch_processing", "parallel_execution", "job_scheduling"],
            "dependencies": ["ml_integration", "resource_manager"],
            "resource_requirements": {"memory": "2GB", "cpu": "4 cores"}
        },
        "production_registry": {
            "description": "MLflow model versioning and registry",
            "file_path": "ml/models/production_registry.py",
            "capabilities": ["model_versioning", "model_registry", "deployment_tracking"],
            "dependencies": ["ml_integration"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core", "disk": "1GB"}
        },
        "context_learner": {
            "description": "Refactored context-specific learning algorithms",
            "file_path": "ml/learning/algorithms/context_learner.py",
            "capabilities": ["contextual_learning", "adaptive_learning", "context_detection", "feature_extraction", "clustering"],
            "dependencies": ["training_data_loader", "domain_detector"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "clustering_optimizer": {
            "description": "High-dimensional clustering optimization",
            "file_path": "ml/optimization/algorithms/clustering_optimizer.py",
            "capabilities": ["clustering", "dimensionality_reduction", "optimization"],
            "dependencies": ["dimensionality_reducer"],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "failure_analyzer": {
            "description": "Failure pattern analysis and recovery with orchestrator integration",
            "file_path": "ml/learning/algorithms/failure_analyzer.py",
            "capabilities": ["failure_detection", "pattern_analysis", "recovery_strategies", "orchestrator_compatible", "robustness_testing"],
            "dependencies": ["ml_integration"],
            "local_config": {
                "data_path": "./data/failure_analysis",
                "output_path": "./outputs/failure_analysis",
                "max_memory_mb": 1024,
                "enable_robustness_validation": True,
                "prometheus_monitoring": True
            },
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "dimensionality_reducer": {
            "description": "Advanced dimensionality reduction with 2025 neural network capabilities",
            "file_path": "ml/optimization/algorithms/dimensionality_reducer.py",
            "capabilities": [
                "pca", "tsne", "umap", "feature_selection",
                "neural_autoencoder", "variational_autoencoder",
                "transformer_attention", "diffusion_models",
                "gpu_acceleration", "incremental_learning"
            ],
            "dependencies": [],
            "resource_requirements": {"memory": "4GB", "cpu": "4 cores", "gpu": "optional"},
            "neural_capabilities": {
                "pytorch_support": True,
                "tensorflow_support": True,
                "gpu_acceleration": True,
                "model_types": ["autoencoder", "vae", "transformer", "diffusion"]
            }
        },
        "synthetic_data_generator": {
            "description": "Production synthetic data generator with modern generative models",
            "file_path": "ml/preprocessing/synthetic_data_generator.py",
            "capabilities": [
                "statistical_generation", "neural_generation", "hybrid_generation",
                "gan_synthesis", "vae_synthesis", "diffusion_synthesis",
                "quality_assessment", "domain_specific_generation"
            ],
            "dependencies": [],
            "resource_requirements": {"memory": "3GB", "cpu": "4 cores", "gpu": "optional"},
            "neural_capabilities": {
                "pytorch_support": True,
                "model_types": ["gan", "vae", "diffusion"],
                "generation_methods": ["statistical", "neural", "hybrid", "diffusion"]
            }
        }
    })
    
    # Tier 2: Optimization & Learning Components (8 components)
    tier2_optimization_components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "insight_engine": {
            "description": "Causal discovery and insights generation with orchestrator integration",
            "file_path": "ml/learning/algorithms/insight_engine.py",
            "capabilities": ["causal_discovery", "insight_generation", "correlation_analysis", "orchestrator_compatible"],
            "dependencies": ["statistical_analyzer"],
            "local_config": {
                "data_path": "./data/insights",
                "output_path": "./outputs/insights",
                "max_memory_mb": 1024,
                "enable_causal_discovery": True
            },
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "rule_analyzer": {
            "description": "Bayesian modeling for rule analysis",
            "file_path": "ml/learning/algorithms/rule_analyzer.py",
            "capabilities": ["bayesian_modeling", "rule_evaluation", "uncertainty_quantification"],
            "dependencies": ["rule_optimizer"],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "context_aware_weighter": {
            "description": "Context-aware feature weighting",
            "file_path": "ml/learning/algorithms/context_aware_weighter.py",
            "capabilities": ["feature_weighting", "context_adaptation", "dynamic_weighting"],
            "dependencies": ["context_learner", "domain_detector"],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"}
        },
        "optimization_validator": {
            "description": "Optimization process validation",
            "file_path": "ml/optimization/validation/optimization_validator.py",
            "capabilities": ["validation", "optimization_testing", "performance_verification"],
            "dependencies": ["rule_optimizer"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "advanced_pattern_discovery": {
            "description": "Advanced pattern mining algorithms",
            "file_path": "ml/learning/patterns/advanced_pattern_discovery.py",
            "capabilities": ["pattern_discovery", "anomaly_detection", "trend_analysis"],
            "dependencies": ["apriori_analyzer"],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "llm_transformer": {
            "description": "LLM-based transformations and enhancements",
            "file_path": "ml/preprocessing/llm_transformer.py",
            "capabilities": ["text_transformation", "prompt_enhancement", "llm_integration"],
            "dependencies": ["ml_integration"],
            "resource_requirements": {"memory": "4GB", "cpu": "2 cores", "gpu": "recommended"}
        },
        "automl_orchestrator": {
            "description": "AutoML coordination and management (existing)",
            "file_path": "ml/automl/orchestrator.py",
            "capabilities": ["automl", "hyperparameter_tuning", "model_selection"],
            "dependencies": ["rule_optimizer", "experiment_orchestrator"],
            "resource_requirements": {"memory": "2GB", "cpu": "4 cores"}
        },
        "context_learner": {
            "description": "Context-specific learning engine with orchestrator integration",
            "file_path": "ml/learning/algorithms/context_learner.py",
            "capabilities": ["context_learning", "pattern_recognition", "adaptive_clustering", "orchestrator_compatible"],
            "dependencies": ["feature_extractor", "clustering_engine"],
            "local_config": {
                "data_path": "./data/context_learning",
                "output_path": "./outputs/context_learning",
                "max_memory_mb": 1024,
                "enable_advanced_clustering": True,
                "cache_enabled": True
            },
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "enhanced_quality_scorer": {
            "description": "Multi-dimensional quality assessment with orchestrator integration",
            "file_path": "ml/learning/quality/enhanced_scorer.py",
            "capabilities": ["quality_assessment", "multi_dimensional_scoring", "statistical_validation", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/quality_assessment",
                "output_path": "./outputs/quality_assessment",
                "max_memory_mb": 512,
                "confidence_level": 0.95,
                "assessment_type": "comprehensive"
            },
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "enhanced_structural_analyzer": {
            "description": "2025 enhanced structural analyzer with graph-based analysis and semantic understanding",
            "file_path": "ml/evaluation/structural_analyzer.py",
            "capabilities": ["structural_analysis", "graph_analysis", "semantic_understanding", "pattern_discovery", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/structural_analysis",
                "output_path": "./outputs/structural_analysis",
                "max_memory_mb": 1024,
                "enable_semantic_analysis": True,
                "enable_graph_analysis": True,
                "enable_pattern_discovery": True,
                "enable_quality_assessment": True,
                "semantic_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "multiarmed_bandit_framework": {
            "description": "Advanced multi-armed bandit framework with orchestrator integration and 2025 best practices",
            "file_path": "ml/optimization/algorithms/multi_armed_bandit.py",
            "capabilities": ["bandit_optimization", "contextual_bandits", "thompson_sampling", "ucb", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/bandit_optimization",
                "output_path": "./outputs/bandit_optimization",
                "max_memory_mb": 512,
                "default_algorithm": "thompson_sampling",
                "warmup_trials": 10,
                "epsilon": 0.1,
                "ucb_confidence": 2.0
            },
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "clustering_optimizer": {
            "description": "Advanced clustering optimizer with orchestrator integration and 2025 best practices",
            "file_path": "ml/optimization/algorithms/clustering_optimizer.py",
            "capabilities": ["clustering_optimization", "umap_reduction", "hdbscan_clustering", "adaptive_parameters", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/clustering_optimization",
                "output_path": "./outputs/clustering_optimization",
                "max_memory_mb": 2048,
                "target_dimensions": 2,
                "umap_n_neighbors": 15,
                "hdbscan_min_cluster_size": 5,
                "memory_efficient_mode": True
            },
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "advanced_early_stopping_framework": {
            "description": "Advanced early stopping framework with orchestrator integration and 2025 best practices",
            "file_path": "ml/optimization/algorithms/early_stopping.py",
            "capabilities": ["early_stopping", "group_sequential_design", "alpha_spending", "futility_stopping", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/early_stopping",
                "output_path": "./outputs/early_stopping",
                "max_memory_mb": 512,
                "alpha": 0.05,
                "beta": 0.2,
                "max_looks": 10,
                "alpha_spending_function": "obrien_fleming",
                "enable_futility_stopping": True
            },
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "enhanced_optimization_validator": {
            "description": "Enhanced optimization validator with 2025 best practices including Bayesian validation, robust statistics, and causal inference",
            "file_path": "ml/optimization/validation/optimization_validator.py",
            "capabilities": ["optimization_validation", "bayesian_validation", "robust_statistics", "causal_inference", "uncertainty_quantification", "orchestrator_compatible"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/optimization_validation",
                "output_path": "./outputs/optimization_validation",
                "max_memory_mb": 1024,
                "validation_method": "comprehensive",
                "enable_bayesian_validation": True,
                "enable_causal_inference": True,
                "enable_robust_methods": True,
                "bootstrap_samples": 10000,
                "permutation_samples": 10000,
                "significance_level": 0.05,
                "min_effect_size": 0.2
            },
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "automl_callbacks": {
            "description": "ML optimization callbacks and hooks",
            "file_path": "ml/automl/callbacks.py",
            "capabilities": ["callbacks", "hooks", "event_handling"],
            "dependencies": ["automl_orchestrator"],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"}
        }
    })
    
    # Tier 3: Evaluation & Analysis Components (10 components)
    tier3_evaluation_components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "experiment_orchestrator": {
            "description": "Experiment management and A/B testing (existing)",
            "file_path": "ml/evaluation/experiment_orchestrator.py",
            "capabilities": ["ab_testing", "experiment_management", "statistical_validation"],
            "dependencies": [],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "advanced_statistical_validator": {
            "description": "Advanced statistical validation with orchestrator integration and 2025 best practices",
            "file_path": "ml/evaluation/advanced_statistical_validator.py",
            "capabilities": ["statistical_testing", "hypothesis_testing", "significance_analysis", "orchestrator_compatible", "multiple_testing_correction"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/statistical_validation",
                "output_path": "./outputs/statistical_validation",
                "max_memory_mb": 1024,
                "alpha": 0.05,
                "power_threshold": 0.8,
                "min_effect_size": 0.1,
                "bootstrap_samples": 10000
            },
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "causal_inference_analyzer": {
            "description": "Advanced causal analysis and inference with orchestrator integration",
            "file_path": "ml/evaluation/causal_inference_analyzer.py",
            "capabilities": ["causal_analysis", "confounding_detection", "treatment_effects", "orchestrator_compatible", "assumption_testing"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/causal_analysis",
                "output_path": "./outputs/causal_analysis",
                "max_memory_mb": 1024,
                "confidence_level": 0.95,
                "enable_robustness_testing": True,
                "enable_sensitivity_analysis": True
            },
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "pattern_significance_analyzer": {
            "description": "Pattern recognition and significance testing with orchestrator integration",
            "file_path": "ml/evaluation/pattern_significance_analyzer.py",
            "capabilities": ["pattern_recognition", "significance_testing", "anomaly_detection", "orchestrator_compatible", "business_insights"],
            "dependencies": [],
            "local_config": {
                "data_path": "./data/pattern_analysis",
                "output_path": "./outputs/pattern_analysis",
                "max_memory_mb": 1024,
                "alpha": 0.05,
                "min_sample_size": 30,
                "effect_size_threshold": 0.1,
                "apply_multiple_testing_correction": True
            },
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "statistical_analyzer": {
            "description": "Core statistical analysis engine",
            "file_path": "ml/evaluation/statistical_analyzer.py",
            "capabilities": ["descriptive_statistics", "inferential_statistics", "regression_analysis"],
            "dependencies": [],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "structural_analyzer": {
            "description": "Prompt structure analysis and optimization",
            "file_path": "ml/evaluation/structural_analyzer.py",
            "capabilities": ["structure_analysis", "syntax_checking", "format_validation"],
            "dependencies": ["linguistic_analyzer"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "domain_feature_extractor": {
            "description": "Feature vector creation and domain analysis",
            "file_path": "ml/analysis/domain_feature_extractor.py",
            "capabilities": ["feature_extraction", "domain_analysis", "vector_generation"],
            "dependencies": ["domain_detector"],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "linguistic_analyzer": {
            "description": "Linguistic analysis and NLP processing",
            "file_path": "ml/analysis/linguistic_analyzer.py",
            "capabilities": ["nlp_processing", "linguistic_analysis", "sentiment_analysis"],
            "dependencies": ["ner_extractor"],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "dependency_parser": {
            "description": "Syntactic analysis and dependency parsing",
            "file_path": "ml/analysis/dependency_parser.py",
            "capabilities": ["syntactic_parsing", "dependency_analysis", "grammar_checking"],
            "dependencies": [],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "domain_detector": {
            "description": "Domain classification and context detection",
            "file_path": "ml/analysis/domain_detector.py",
            "capabilities": ["domain_classification", "context_detection", "topic_modeling"],
            "dependencies": [],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        }
    })
    
    # Tier 4: Performance & Testing Components (8 components)  
    tier4_performance_components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "advanced_ab_testing": {
            "description": "Enhanced A/B testing with advanced analytics",
            "file_path": "performance/testing/advanced_ab_testing.py",
            "capabilities": ["ab_testing", "multivariate_testing", "statistical_power"],
            "dependencies": ["analytics"],
            "resource_requirements": {"memory": "1GB", "cpu": "2 cores"}
        },
        "canary_testing": {
            "description": "Feature rollout and canary testing",
            "file_path": "performance/testing/canary_testing.py",
            "capabilities": ["canary_deployment", "gradual_rollout", "risk_management"],
            "dependencies": ["monitoring"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "real_time_analytics": {
            "description": "Live monitoring and real-time analytics",
            "file_path": "performance/analytics/real_time_analytics.py",
            "capabilities": ["real_time_monitoring", "live_dashboards", "streaming_analytics"],
            "dependencies": ["analytics"],
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores"}
        },
        "analytics": {
            "description": "Rule effectiveness analytics and reporting",
            "file_path": "performance/analytics/analytics.py",
            "capabilities": ["analytics", "reporting", "performance_metrics"],
            "dependencies": [],
            "resource_requirements": {"memory": "1GB", "cpu": "1 core"}
        },
        "monitoring": {
            "description": "Performance monitoring and alerting",
            "file_path": "performance/monitoring/monitoring.py",
            "capabilities": ["performance_monitoring", "alerting", "health_checks"],
            "dependencies": [],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "async_optimizer": {
            "description": "Asynchronous optimization engine",
            "file_path": "performance/optimization/async_optimizer.py",
            "capabilities": ["async_optimization", "parallel_processing", "task_scheduling"],
            "dependencies": ["monitoring"],
            "resource_requirements": {"memory": "1GB", "cpu": "4 cores"}
        },
        "early_stopping": {
            "description": "Early stopping algorithms for training",
            "file_path": "ml/optimization/algorithms/early_stopping.py",
            "capabilities": ["early_stopping", "convergence_detection", "training_optimization"],
            "dependencies": ["rule_optimizer"],
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"}
        },
        "background_manager": {
            "description": "Background task management and coordination",
            "file_path": "performance/monitoring/health/background_manager.py",
            "capabilities": ["background_tasks", "task_coordination", "resource_management"],
            "dependencies": ["monitoring"],
            "resource_requirements": {"memory": "512MB", "cpu": "2 cores"}
        },
        "multi_level_cache": {
            "description": "Advanced multi-level caching system with OpenTelemetry tracing",
            "file_path": "utils/multi_level_cache.py",
            "capabilities": ["l1_cache", "l2_cache", "l3_fallback", "opentelemetry_tracing", "performance_monitoring"],
            "dependencies": [],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"},
            "metadata": {
                "cache_levels": 3,
                "tracing_enabled": True,
                "metrics_export": True
            }
        },
        "resource_manager": {
            "description": "Core ML orchestration resource manager with circuit breakers and Kubernetes integration",
            "file_path": "ml/orchestration/core/resource_manager.py",
            "capabilities": ["cpu_allocation", "gpu_management", "monitoring", "circuit_breakers", "kubernetes_integration"],
            "dependencies": [],
            "resource_requirements": {"memory": "512MB", "cpu": "2 cores"},
            "metadata": {
                "circuit_breakers_enabled": True,
                "kubernetes_integration": True,
                "gpu_aware": True,
                "mig_support": True
            }
        },
        "health_service": {
            "description": "Enhanced health monitoring service with 2025 observability features",
            "file_path": "performance/monitoring/health/service.py",
            "capabilities": [
                "health_monitoring", "circuit_breakers", "predictive_analysis",
                "opentelemetry_tracing", "prometheus_metrics", "dependency_mapping",
                "trend_analysis", "sla_monitoring"
            ],
            "dependencies": ["monitoring"],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"},
            "metadata": {
                "circuit_breakers_enabled": True,
                "predictive_monitoring": True,
                "opentelemetry_integration": True,
                "prometheus_metrics": True,
                "health_caching": True,
                "dependency_graph": True
            }
        },
        "ml_resource_manager_health_checker": {
            "description": "ML resource manager health monitoring with threshold-based alerting",
            "file_path": "performance/monitoring/health/ml_orchestration_checkers.py",
            "capabilities": [
                "resource_health_monitoring", "threshold_alerting", "usage_tracking",
                "critical_resource_detection", "prometheus_instrumentation"
            ],
            "dependencies": ["resource_manager", "health_service"],
            "resource_requirements": {"memory": "128MB", "cpu": "0.5 cores"},
            "metadata": {
                "critical_threshold": 90,
                "warning_threshold": 75,
                "prometheus_metrics": True,
                "real_time_monitoring": True
            }
        },
        "prepared_statement_cache": {
            "description": "Prepared statement caching for query performance optimization",
            "file_path": "database/query_optimizer.py",
            "capabilities": ["cache_performance", "query_optimization", "cache_efficiency", "orchestrator_compatible"],
            "dependencies": [],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"},
            "local_config": {
                "max_cache_size": 100,
                "enable_performance_monitoring": True,
                "cache_hit_threshold": 0.8
            }
        },
        "type_safe_psycopg_client": {
            "description": "Type-safe PostgreSQL client with orchestrator integration",
            "file_path": "database/psycopg_client.py",
            "capabilities": [
                "performance_metrics", "connection_health", "query_analysis",
                "type_safety_validation", "comprehensive_analysis", "orchestrator_compatible",
                "real_time_monitoring", "circuit_breaker_protection", "error_classification"
            ],
            "dependencies": ["database_config"],
            "resource_requirements": {"memory": "512MB", "cpu": "2 cores"},
            "local_config": {
                "target_query_time_ms": 50,
                "target_cache_hit_ratio": 0.9,
                "pool_min_size": 2,
                "pool_max_size": 20,
                "enable_circuit_breaker": True,
                "enable_error_metrics": True,
                "statement_timeout": 30,
                "connection_timeout": 10
            }
        }
    })
    
    # Tier 6: Security & Encryption Components (2 components)
    tier6_security_components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "secure_key_manager": {
            "description": "Enterprise-grade secure key management with rotation and versioning",
            "file_path": "security/key_manager.py",
            "capabilities": ["key_management", "key_rotation", "key_versioning", "secure_storage", "orchestrator_compatible"],
            "dependencies": [],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"},
            "local_config": {
                "key_dir": "./keys",
                "rotation_interval_hours": 24,
                "max_key_age_hours": 72,
                "key_version_limit": 5,
                "auto_rotation_enabled": True
            },
            "metadata": {
                "security_tier": True,
                "encryption_standard": "AES-256",
                "key_derivation": "PBKDF2HMAC-SHA256"
            }
        },
        "fernet_key_manager": {
            "description": "Fernet encryption wrapper using SecureKeyManager for ML data protection",
            "file_path": "security/key_manager.py",
            "capabilities": ["fernet_encryption", "data_encryption", "secure_encryption", "orchestrator_compatible", "round_trip_testing"],
            "dependencies": ["secure_key_manager"],
            "resource_requirements": {"memory": "256MB", "cpu": "1 core"},
            "local_config": {
                "encryption_backend": "fernet",
                "test_data_enabled": True,
                "performance_monitoring": True
            },
            "metadata": {
                "security_tier": True,
                "encryption_type": "symmetric",
                "algorithm": "AES-128-CBC-HMAC-SHA256"
            }
        },
        "robustness_evaluator": {
            "description": "2025 NIST-compliant ML robustness evaluation with comprehensive adversarial testing and security validation",
            "file_path": "security/adversarial_defense.py",
            "capabilities": [
                "adversarial_testing",
                "robustness_metrics",
                "security_validation",
                "threat_detection",
                "nist_compliance",
                "orchestrator_compatible",
                "async_evaluation",
                "multi_attack_testing",
                "defense_effectiveness",
                "vulnerability_assessment"
            ],
            "dependencies": ["adversarial_defense"],
            "local_config": {
                "data_path": "./data/robustness_evaluation",
                "output_path": "./outputs/robustness_evaluation",
                "max_memory_mb": 2048,
                "enable_adversarial_testing": True,
                "enable_nist_compliance": True,
                "enable_async_evaluation": True,
                "attack_types": ["fgsm", "pgd", "cw", "deepfool", "boundary"],
                "robustness_metrics": ["accuracy_drop", "attack_success_rate", "perturbation_distance", "confidence_degradation"],
                "security_thresholds": {
                    "min_robustness_score": 0.7,
                    "max_attack_success_rate": 0.3,
                    "min_defense_effectiveness": 0.8
                },
                "evaluation_modes": ["comprehensive", "fast", "targeted"],
                "threat_detection_enabled": True,
                "vulnerability_scanning": True
            },
            "resource_requirements": {"memory": "2GB", "cpu": "2 cores", "gpu": "optional"},
            "metadata": {
                "security_tier": True,
                "nist_compliant": True,
                "evaluation_framework": "2025",
                "threat_model": "comprehensive"
            }
        },
        "prompt_data_protection": {
            "description": "2025 GDPR-compliant prompt data protection with differential privacy, async operations, and ML pipeline integration",
            "file_path": "core/services/security.py",
            "capabilities": [
                "data_protection",
                "sensitive_data_detection",
                "prompt_sanitization",
                "gdpr_compliance",
                "differential_privacy",
                "audit_logging",
                "performance_monitoring",
                "real_time_processing",
                "risk_assessment",
                "privacy_by_design",
                "orchestrator_compatible",
                "batch_processing",
                "async_operations",
                "compliance_scoring",
                "pattern_detection"
            ],
            "dependencies": ["analytics", "database"],
            "local_config": {
                "data_path": "./data/security",
                "output_path": "./outputs/security_audit",
                "max_memory_mb": 512,
                "enable_gdpr_compliance": True,
                "enable_differential_privacy": True,
                "enable_audit_logging": True,
                "data_retention_days": 30,
                "max_processing_time_ms": 1000,
                "pattern_confidence_threshold": 0.7,
                "privacy_techniques": ["redaction", "masking", "tokenization", "differential_privacy"],
                "risk_levels": ["LOW", "MEDIUM", "HIGH"],
                "compliance_framework": "2025.1",
                "performance_monitoring": True,
                "real_time_processing": True,
                "batch_size_limit": 100
            },
            "resource_requirements": {"memory": "512MB", "cpu": "1 core"},
            "metadata": {
                "security_tier": True,
                "gdpr_compliant": True,
                "privacy_framework": "2025",
                "data_protection_level": "enterprise",
                "async_compatible": True,
                "real_time_capable": True,
                "batch_processing_capable": True,
                "compliance_standards": ["GDPR", "CCPA", "NIST"],
                "privacy_techniques": ["differential_privacy", "data_masking", "tokenization"],
                "audit_capabilities": ["comprehensive_logging", "compliance_reporting", "risk_assessment"]
            }
        }
    })
    
    def get_all_component_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get all component definitions across all tiers."""
        all_components = {}
        all_components.update(self.tier1_core_components)
        all_components.update(self.tier2_optimization_components)
        all_components.update(self.tier3_evaluation_components)
        all_components.update(self.tier4_performance_components)
        all_components.update(self.tier6_security_components)
        return all_components
    
    def get_tier_components(self, tier: ComponentTier) -> Dict[str, Dict[str, Any]]:
        """Get components for a specific tier."""
        if tier == ComponentTier.TIER_1_CORE:
            return self.tier1_core_components
        elif tier == ComponentTier.TIER_2_OPTIMIZATION:
            return self.tier2_optimization_components
        elif tier == ComponentTier.TIER_3_EVALUATION:
            return self.tier3_evaluation_components
        elif tier == ComponentTier.TIER_4_PERFORMANCE:
            return self.tier4_performance_components
        elif tier == ComponentTier.TIER_6_SECURITY:
            return self.tier6_security_components
        else:
            return {}  # Other tiers not yet implemented
    
    def create_component_info(self, name: str, definition: Dict[str, Any], tier: ComponentTier) -> ComponentInfo:
        """Create ComponentInfo from definition."""
        capabilities = []
        for cap_name in definition.get("capabilities", []):
            capabilities.append(ComponentCapability(
                name=cap_name,
                description=f"{cap_name} capability",
                input_types=["data"],
                output_types=["result"]
            ))
        
        return ComponentInfo(
            name=name,
            tier=tier,
            description=definition.get("description", ""),
            version="1.0.0",
            capabilities=capabilities,
            dependencies=definition.get("dependencies", []),
            resource_requirements=definition.get("resource_requirements", {}),
            metadata={
                "file_path": definition.get("file_path", ""),
                "implementation_status": "phase1_registered"
            }
        )