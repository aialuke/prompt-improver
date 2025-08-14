"""Test Data Factory for Real Behavior Testing.

Generates realistic test data for comprehensive testing scenarios across all decomposed services.
Provides domain-specific data generation for ML, database, API, and validation testing.
"""

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class TestDataset:
    """Structured test dataset for various testing scenarios."""
    
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    size: str = "small"  # small, medium, large
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Dataset content
    data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    record_count: int = 0
    domains: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


class TestDataFactory:
    """Factory for generating comprehensive test data for real behavior testing."""
    
    def __init__(self):
        """Initialize test data factory."""
        self.factory_id = str(uuid.uuid4())[:8]
        logger.info(f"TestDataFactory initialized: {self.factory_id}")
    
    async def create_ml_test_dataset(
        self,
        size: str = "small",
        domains: List[str] = None,
        include_effectiveness_scores: bool = True,
        include_context: bool = True
    ) -> TestDataset:
        """Create ML test dataset for intelligence services testing.
        
        Args:
            size: Dataset size ("small", "medium", "large")
            domains: List of domains to include
            include_effectiveness_scores: Include effectiveness scoring data
            include_context: Include contextual information
            
        Returns:
            TestDataset for ML testing
        """
        domains = domains or ["general", "technical", "creative"]
        
        # Size configuration
        size_config = {
            "small": {"sessions": 100, "rules": 10, "patterns": 20},
            "medium": {"sessions": 500, "rules": 25, "patterns": 50},
            "large": {"sessions": 2000, "rules": 50, "patterns": 100},
        }
        config = size_config.get(size, size_config["small"])
        
        dataset = TestDataset(
            name=f"ml_test_dataset_{size}",
            description=f"ML intelligence services test dataset ({size})",
            size=size,
            domains=domains,
            categories=["prompts", "improvements", "sessions", "rules"]
        )
        
        # Generate test data
        test_data = []
        
        # Generate rule data
        rules = self._generate_rule_data(config["rules"], domains)
        test_data.extend(rules)
        
        # Generate session data
        sessions = await self._generate_session_data(
            config["sessions"],
            domains,
            include_effectiveness_scores,
            include_context
        )
        test_data.extend(sessions)
        
        # Generate pattern data
        patterns = self._generate_pattern_data(config["patterns"], domains)
        test_data.extend(patterns)
        
        dataset.data = test_data
        dataset.record_count = len(test_data)
        dataset.metadata = {
            "rules_count": len(rules),
            "sessions_count": len(sessions),
            "patterns_count": len(patterns),
            "domains": domains,
            "size_config": config,
        }
        
        logger.info(f"Created ML test dataset: {dataset.record_count} records across {len(domains)} domains")
        
        return dataset
    
    def _generate_rule_data(self, count: int, domains: List[str]) -> List[Dict[str, Any]]:
        """Generate rule data for ML intelligence testing."""
        rules = []
        
        rule_templates = {
            "general": [
                "Improve clarity and conciseness",
                "Add specific examples and context",
                "Enhance structure and organization",
                "Include relevant background information",
                "Use more precise language",
            ],
            "technical": [
                "Add technical specifications and requirements",
                "Include error handling and edge cases",
                "Provide implementation details and examples",
                "Add performance considerations",
                "Include security and validation requirements",
            ],
            "creative": [
                "Enhance descriptive language and imagery",
                "Add emotional depth and connection",
                "Improve narrative flow and pacing",
                "Include sensory details and atmosphere",
                "Develop character motivation and conflict",
            ],
        }
        
        for i in range(count):
            domain = domains[i % len(domains)]
            templates = rule_templates.get(domain, rule_templates["general"])
            
            rule_id = f"rule_{domain}_{i:03d}"
            rule_text = templates[i % len(templates)]
            
            rule = {
                "type": "rule",
                "rule_id": rule_id,
                "rule_text": rule_text,
                "domain": domain,
                "category": "improvement_rule",
                "effectiveness_score": random.uniform(0.6, 0.95),
                "usage_count": random.randint(10, 200),
                "success_rate": random.uniform(0.7, 0.9),
                "created_at": (datetime.now(timezone.utc) - timedelta(days=random.randint(1, 365))).isoformat(),
                "metadata": {
                    "complexity": random.choice(["simple", "moderate", "complex"]),
                    "applicability": random.choice(["general", "specific", "contextual"]),
                    "priority": random.randint(1, 5),
                }
            }
            
            rules.append(rule)
        
        return rules
    
    async def _generate_session_data(
        self,
        count: int,
        domains: List[str],
        include_effectiveness: bool,
        include_context: bool
    ) -> List[Dict[str, Any]]:
        """Generate session data for ML intelligence testing."""
        sessions = []
        
        prompt_templates = {
            "general": [
                "Write a summary about {topic}",
                "Explain the importance of {topic}",
                "Describe the benefits of {topic}",
                "Compare different approaches to {topic}",
                "Provide an overview of {topic}",
            ],
            "technical": [
                "Implement a {topic} algorithm in Python",
                "Design a {topic} architecture for scalability",
                "Debug issues with {topic} implementation",
                "Optimize {topic} performance for large datasets",
                "Create comprehensive tests for {topic} functionality",
            ],
            "creative": [
                "Write a story featuring {topic}",
                "Create a compelling narrative about {topic}",
                "Develop characters dealing with {topic}",
                "Describe the atmosphere of {topic}",
                "Craft dialogue that reveals {topic}",
            ],
        }
        
        topics = {
            "general": ["teamwork", "innovation", "leadership", "communication", "problem-solving"],
            "technical": ["microservices", "machine learning", "database optimization", "API design", "security"],
            "creative": ["adventure", "mystery", "friendship", "discovery", "transformation"],
        }
        
        for i in range(count):
            domain = domains[i % len(domains)]
            session_id = f"session_{domain}_{i:05d}"
            
            # Select templates and topics
            templates = prompt_templates.get(domain, prompt_templates["general"])
            topic_list = topics.get(domain, topics["general"])
            
            prompt_template = templates[i % len(templates)]
            topic = topic_list[i % len(topic_list)]
            
            original_prompt = prompt_template.format(topic=topic)
            
            # Generate improved prompt
            improvements = [
                f"Enhanced {original_prompt.lower()} with specific examples",
                f"Detailed {original_prompt.lower()} including step-by-step guidance",
                f"Comprehensive {original_prompt.lower()} with relevant context",
                f"Structured {original_prompt.lower()} with clear objectives",
            ]
            improved_prompt = improvements[i % len(improvements)]
            
            session = {
                "type": "session",
                "session_id": session_id,
                "original_prompt": original_prompt,
                "improved_prompt": improved_prompt,
                "domain": domain,
                "topic": topic,
                "created_at": (datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 8760))).isoformat(),
                "rules_applied": [f"rule_{domain}_{random.randint(0, 9):03d}" for _ in range(random.randint(1, 3))],
            }
            
            if include_effectiveness:
                session.update({
                    "effectiveness_score": random.uniform(0.65, 0.95),
                    "improvement_rating": random.uniform(0.7, 0.9),
                    "user_satisfaction": random.uniform(0.8, 1.0),
                })
            
            if include_context:
                session.update({
                    "context": {
                        "user_level": random.choice(["beginner", "intermediate", "advanced"]),
                        "use_case": random.choice(["learning", "professional", "personal", "academic"]),
                        "complexity_preference": random.choice(["simple", "detailed", "comprehensive"]),
                        "style_preference": random.choice(["formal", "casual", "technical", "creative"]),
                    },
                    "metadata": {
                        "processing_time_ms": random.uniform(50, 500),
                        "tokens_original": len(original_prompt.split()) * random.randint(1, 2),
                        "tokens_improved": len(improved_prompt.split()) * random.randint(1, 2),
                        "improvement_type": random.choice(["clarity", "detail", "structure", "examples"]),
                    }
                })
            
            sessions.append(session)
        
        return sessions
    
    def _generate_pattern_data(self, count: int, domains: List[str]) -> List[Dict[str, Any]]:
        """Generate pattern data for pattern discovery testing."""
        patterns = []
        
        pattern_types = ["structural", "linguistic", "contextual", "behavioral", "semantic"]
        
        for i in range(count):
            domain = domains[i % len(domains)]
            pattern_id = f"pattern_{domain}_{i:03d}"
            
            pattern = {
                "type": "pattern",
                "pattern_id": pattern_id,
                "pattern_type": pattern_types[i % len(pattern_types)],
                "domain": domain,
                "strength": random.uniform(0.3, 0.95),
                "frequency": random.randint(5, 100),
                "support": random.uniform(0.1, 0.8),
                "confidence": random.uniform(0.6, 0.95),
                "description": f"Pattern {i:03d} in {domain} domain showing {pattern_types[i % len(pattern_types)]} characteristics",
                "discovered_at": (datetime.now(timezone.utc) - timedelta(days=random.randint(1, 90))).isoformat(),
                "examples": [
                    f"Example {j+1} for pattern {pattern_id}"
                    for j in range(random.randint(2, 5))
                ],
                "metadata": {
                    "algorithm": "hdbscan",
                    "cluster_size": random.randint(5, 50),
                    "stability": random.uniform(0.5, 0.95),
                    "outlier_ratio": random.uniform(0.05, 0.3),
                }
            }
            
            patterns.append(pattern)
        
        return patterns
    
    async def create_comprehensive_rule_dataset(
        self,
        rule_count: int = 20,
        session_count_per_rule: int = 50,
        domains: List[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive rule dataset for rule analysis testing.
        
        Args:
            rule_count: Number of rules to generate
            session_count_per_rule: Sessions per rule
            domains: List of domains
            
        Returns:
            Comprehensive rule dataset
        """
        domains = domains or ["general", "technical", "creative"]
        
        # Generate rules
        rules = self._generate_rule_data(rule_count, domains)
        rule_ids = [rule["rule_id"] for rule in rules]
        
        # Generate sessions for each rule
        sessions_by_rule = {}
        all_sessions = []
        
        for rule in rules:
            rule_id = rule["rule_id"]
            domain = rule["domain"]
            
            # Generate sessions for this rule
            rule_sessions = await self._generate_session_data(
                session_count_per_rule,
                [domain],  # Use rule's domain
                include_effectiveness=True,
                include_context=True
            )
            
            # Associate sessions with rule
            for session in rule_sessions:
                session["primary_rule"] = rule_id
                session["rule_effectiveness"] = rule["effectiveness_score"]
            
            sessions_by_rule[rule_id] = rule_sessions
            all_sessions.extend(rule_sessions)
        
        return {
            "rule_ids": rule_ids,
            "rules": rules,
            "sessions_by_rule": sessions_by_rule,
            "all_sessions": all_sessions,
            "metadata": {
                "rule_count": rule_count,
                "total_sessions": len(all_sessions),
                "sessions_per_rule": session_count_per_rule,
                "domains": domains,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }
    
    async def create_batch_processing_data(
        self,
        batch_size: int = 100,
        operation_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Create batch processing data for batch service testing.
        
        Args:
            batch_size: Size of the batch
            operation_types: Types of operations to include
            
        Returns:
            List of batch processing items
        """
        operation_types = operation_types or ["rule_analysis", "pattern_discovery", "prediction"]
        
        batch_data = []
        
        for i in range(batch_size):
            operation_type = operation_types[i % len(operation_types)]
            
            item = {
                "batch_item_id": f"batch_item_{i:05d}",
                "operation_type": operation_type,
                "priority": random.randint(1, 5),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "estimated_processing_time_ms": random.uniform(10, 200),
            }
            
            # Add operation-specific data
            if operation_type == "rule_analysis":
                item.update({
                    "rule_id": f"rule_{random.randint(1, 50):03d}",
                    "analysis_type": random.choice(["effectiveness", "usage", "combination"]),
                    "data_range": {
                        "start_date": (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                        "end_date": datetime.now(timezone.utc).isoformat(),
                    }
                })
            
            elif operation_type == "pattern_discovery":
                item.update({
                    "data_source": random.choice(["sessions", "feedback", "usage_logs"]),
                    "pattern_types": random.sample(["structural", "linguistic", "contextual"], 2),
                    "min_support": random.uniform(0.1, 0.3),
                    "min_confidence": random.uniform(0.6, 0.8),
                })
            
            elif operation_type == "prediction":
                item.update({
                    "prediction_target": random.choice(["effectiveness", "usage", "satisfaction"]),
                    "input_features": random.sample(["domain", "complexity", "context", "history"], 3),
                    "model_type": random.choice(["classification", "regression", "clustering"]),
                })
            
            batch_data.append(item)
        
        return batch_data
    
    def create_error_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create error test scenarios for error handling services.
        
        Returns:
            List of error test scenarios
        """
        scenarios = []
        
        # Database error scenarios
        database_scenarios = [
            {
                "category": "database",
                "error_type": "ConnectionError",
                "error_message": "Connection to database failed: Connection refused",
                "operation": "database_connection",
                "context": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db",
                    "connection_timeout": 30,
                },
                "expected_retryable": True,
                "expected_category": "connection_error",
            },
            {
                "category": "database",
                "error_type": "RuntimeError",
                "error_message": "relation 'missing_table' does not exist",
                "operation": "query_execution",
                "context": {
                    "sql": "SELECT * FROM missing_table",
                    "parameters": [],
                },
                "expected_retryable": False,
                "expected_category": "schema_error",
            },
            {
                "category": "database",
                "error_type": "TimeoutError",
                "error_message": "Query timeout: execution exceeded 30 seconds",
                "operation": "long_running_query",
                "context": {
                    "sql": "SELECT * FROM large_table ORDER BY complex_calculation",
                    "timeout_seconds": 30,
                },
                "expected_retryable": True,
                "expected_category": "performance_error",
            },
        ]
        
        # Network error scenarios
        network_scenarios = [
            {
                "category": "network",
                "error_type": "TimeoutError",
                "error_message": "HTTP request timeout after 30 seconds",
                "operation": "api_request",
                "context": {
                    "url": "https://api.example.com/slow-endpoint",
                    "method": "GET",
                    "timeout_ms": 30000,
                },
                "expected_retryable": True,
                "expected_category": "timeout_error",
            },
            {
                "category": "network",
                "error_type": "RuntimeError",
                "error_message": "HTTP 503: Service Unavailable",
                "operation": "api_request",
                "context": {
                    "url": "https://api.example.com/endpoint",
                    "method": "POST",
                    "status_code": 503,
                },
                "expected_retryable": True,
                "expected_category": "server_error",
            },
            {
                "category": "network",
                "error_type": "ValueError",
                "error_message": "HTTP 400: Bad Request - Invalid JSON payload",
                "operation": "api_request",
                "context": {
                    "url": "https://api.example.com/endpoint",
                    "method": "POST",
                    "status_code": 400,
                    "payload": "invalid json",
                },
                "expected_retryable": False,
                "expected_category": "client_error",
            },
        ]
        
        # Validation error scenarios
        validation_scenarios = [
            {
                "category": "validation",
                "error_type": "ValueError",
                "error_message": "Field 'email' is required but not provided",
                "operation": "input_validation",
                "context": {
                    "input_data": {"name": "John", "age": 25},
                    "required_fields": ["name", "email", "age"],
                },
                "expected_retryable": False,
                "expected_category": "required_field_error",
            },
            {
                "category": "validation",
                "error_type": "ValueError",
                "error_message": "Potential SQL injection detected in input",
                "operation": "security_validation",
                "context": {
                    "input_data": {"query": "'; DROP TABLE users; --"},
                    "threat_detected": True,
                },
                "expected_retryable": False,
                "expected_category": "security_violation",
            },
        ]
        
        scenarios.extend(database_scenarios)
        scenarios.extend(network_scenarios)
        scenarios.extend(validation_scenarios)
        
        # Add unique IDs and metadata
        for i, scenario in enumerate(scenarios):
            scenario.update({
                "scenario_id": f"error_scenario_{i:03d}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "test_priority": random.randint(1, 3),
            })
        
        return scenarios
    
    def create_performance_test_data(self, size: str = "medium") -> Dict[str, Any]:
        """Create performance test data for benchmarking.
        
        Args:
            size: Size of test data ("small", "medium", "large")
            
        Returns:
            Performance test data
        """
        size_config = {
            "small": {"operations": 100, "concurrent": 5, "data_size_kb": 1},
            "medium": {"operations": 1000, "concurrent": 20, "data_size_kb": 10},
            "large": {"operations": 10000, "concurrent": 50, "data_size_kb": 100},
        }
        config = size_config.get(size, size_config["medium"])
        
        # Generate test operations
        operations = []
        for i in range(config["operations"]):
            operation = {
                "operation_id": f"perf_op_{i:06d}",
                "operation_type": random.choice(["read", "write", "update", "delete", "complex_query"]),
                "payload_size_bytes": random.randint(100, config["data_size_kb"] * 1024),
                "expected_duration_ms": random.uniform(1, 100),
                "priority": random.randint(1, 5),
                "concurrent_group": i % config["concurrent"],
            }
            operations.append(operation)
        
        return {
            "size": size,
            "config": config,
            "operations": operations,
            "metadata": {
                "total_operations": len(operations),
                "concurrent_groups": config["concurrent"],
                "avg_payload_size": sum(op["payload_size_bytes"] for op in operations) / len(operations),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        }