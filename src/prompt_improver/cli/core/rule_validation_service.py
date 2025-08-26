"""Rule Validation Service for Week 4 Smart Initialization
Validates existing seeded rules, loads rule metadata, and ensures rule parameter integrity.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.database.models import RuleMetadata
from prompt_improver.shared.interfaces.protocols.database import SessionManagerProtocol


class RuleValidationService:
    """Comprehensive rule validation service implementing 2025 best practices.

    Features:
    - Seeded rule validation and integrity checking
    - Rule metadata loading and parameter validation
    - Performance history analysis
    - Rule configuration validation
    - Intelligent rule recommendations
    """

    def __init__(self, session_manager: SessionManagerProtocol) -> None:
        self.session_manager = session_manager
        self.logger = logging.getLogger("apes.rule_validation")
        self.expected_rules = {
            "clarity_enhancement": {
                "category": "fundamental",
                "min_priority": 8,
                "required_parameters": ["min_clarity_score", "use_structured_xml"],
            },
            "chain_of_thought": {
                "category": "reasoning",
                "min_priority": 7,
                "required_parameters": ["enable_step_by_step", "use_thinking_tags"],
            },
            "few_shot_examples": {
                "category": "examples",
                "min_priority": 6,
                "required_parameters": [
                    "optimal_example_count",
                    "require_diverse_examples",
                ],
            },
            "context_optimization": {
                "category": "context",
                "min_priority": 5,
                "required_parameters": ["context_placement", "relevance_threshold"],
            },
            "output_formatting": {
                "category": "formatting",
                "min_priority": 4,
                "required_parameters": ["structured_output", "format_consistency"],
            },
            "specificity_enhancement": {
                "category": "fundamental",
                "min_priority": 3,
                "required_parameters": [
                    "vague_language_threshold",
                    "require_specific_outcomes",
                ],
            },
        }

    async def validate_all_rules(self) -> dict[str, Any]:
        """Comprehensive validation of all seeded rules and metadata.

        Returns:
            Complete validation report with recommendations
        """
        validation_start = datetime.now(UTC)
        validation_report = {
            "validation_timestamp": validation_start.isoformat(),
            "overall_status": "unknown",
            "rule_count": {
                "expected": len(self.expected_rules),
                "found": 0,
                "valid": 0,
            },
            "seeded_rules": {},
            "metadata_validation": {},
            "performance_analysis": {},
            "recommendations": [],
            "validation_time_ms": 0,
        }
        try:
            session_manager = self.session_manager
            async with session_manager.get_async_session() as db_session:
                rule_existence = await self._validate_rule_existence(db_session)
                validation_report["rule_count"]["found"] = rule_existence["total_rules"]
                validation_report["seeded_rules"] = rule_existence["rules"]
                metadata_validation = await self._validate_rule_metadata(db_session)
                validation_report["metadata_validation"] = metadata_validation
                parameter_validation = await self._validate_rule_parameters(db_session)
                validation_report["parameter_validation"] = parameter_validation
                performance_analysis = await self._analyze_rule_performance(db_session)
                validation_report["performance_analysis"] = performance_analysis
                recommendations = await self._generate_rule_recommendations(
                    rule_existence,
                    metadata_validation,
                    parameter_validation,
                    performance_analysis,
                )
                validation_report["recommendations"] = recommendations
                valid_rules = sum(
                    1
                    for rule in validation_report["seeded_rules"].values()
                    if rule["status"] == "valid"
                )
                validation_report["rule_count"]["valid"] = valid_rules
                if valid_rules >= len(self.expected_rules):
                    validation_report["overall_status"] = "healthy"
                elif valid_rules >= len(self.expected_rules) * 0.8:
                    validation_report["overall_status"] = "acceptable"
                else:
                    validation_report["overall_status"] = "needs_attention"
        except Exception as e:
            self.logger.exception(f"Rule validation failed: {e}")
            validation_report["overall_status"] = "error"
            validation_report["error"] = str(e)
        validation_end = datetime.now(UTC)
        validation_report["validation_time_ms"] = (
            validation_end - validation_start
        ).total_seconds() * 1000
        return validation_report

    async def _validate_rule_existence(
        self, db_session: AsyncSession
    ) -> dict[str, Any]:
        """Validate that expected seeded rules exist in the database."""
        result = await db_session.execute(select(RuleMetadata))
        db_rules = {}
        for rule in result.scalars().all():
            db_rules[rule.rule_id] = {
                "rule_name": rule.rule_name,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "category": rule.category,
            }
        rule_validation = {}
        for rule_id, expected in self.expected_rules.items():
            if rule_id in db_rules:
                db_rule = db_rules[rule_id]
                issues = []
                if not db_rule["enabled"]:
                    issues.append("Rule is disabled")
                if db_rule["priority"] < expected["min_priority"]:
                    issues.append(
                        f"Priority too low: {db_rule['priority']} < {expected['min_priority']}"
                    )
                if db_rule["category"] != expected["category"]:
                    issues.append(
                        f"Category mismatch: {db_rule['category']} != {expected['category']}"
                    )
                rule_validation[rule_id] = {
                    "status": "valid" if not issues else "issues",
                    "found": True,
                    "issues": issues,
                    "details": db_rule,
                }
            else:
                rule_validation[rule_id] = {
                    "status": "missing",
                    "found": False,
                    "issues": ["Rule not found in database"],
                    "details": {},
                }
        return {
            "total_rules": len(db_rules),
            "expected_rules": len(self.expected_rules),
            "rules": rule_validation,
        }

    async def _validate_rule_metadata(self, db_session: AsyncSession) -> dict[str, Any]:
        """Validate rule metadata integrity and completeness."""
        metadata_issues = []
        rule_metadata = {}
        result = await db_session.execute(select(RuleMetadata))
        for rule in result.scalars().all():
            rule_id = rule.rule_id
            default_params = rule.default_parameters
            constraints = rule.parameter_constraints
            version = rule.rule_version
            created_at = rule.created_at
            updated_at = rule.updated_at
            rule_issues = []
            if default_params:
                try:
                    if isinstance(default_params, str):
                        params = json.loads(default_params)
                    else:
                        params = default_params
                    if rule_id in self.expected_rules:
                        required_params = self.expected_rules[rule_id][
                            "required_parameters"
                        ]
                        missing_params = [p for p in required_params if p not in params]
                        if missing_params:
                            rule_issues.append(
                                f"Missing required parameters: {missing_params}"
                            )
                except (json.JSONDecodeError, TypeError) as e:
                    rule_issues.append(f"Invalid default_parameters JSON: {e}")
            else:
                rule_issues.append("No default parameters defined")
            if constraints:
                try:
                    if isinstance(constraints, str):
                        json.loads(constraints)
                except (json.JSONDecodeError, TypeError) as e:
                    rule_issues.append(f"Invalid parameter_constraints JSON: {e}")
            if not version or not version.count(".") >= 2:
                rule_issues.append(f"Invalid version format: {version}")
            rule_metadata[rule_id] = {
                "issues": rule_issues,
                "status": "valid" if not rule_issues else "issues",
                "version": version,
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            }
            metadata_issues.extend(rule_issues)
        return {
            "overall_status": "healthy" if not metadata_issues else "issues_found",
            "total_issues": len(metadata_issues),
            "rule_metadata": rule_metadata,
            "issues": metadata_issues,
        }

    async def _validate_rule_parameters(
        self, db_session: AsyncSession
    ) -> dict[str, Any]:
        """Validate rule parameter configurations and constraints."""
        parameter_validation = {}
        result = await db_session.execute(select(RuleMetadata))
        for rule in result.scalars().all():
            rule_id = rule.rule_id
            default_params = rule.default_parameters
            constraints = rule.parameter_constraints
            validation_result = {
                "status": "valid",
                "issues": [],
                "parameter_count": 0,
                "constraint_violations": [],
            }
            try:
                if default_params:
                    if isinstance(default_params, str):
                        params = json.loads(default_params)
                    else:
                        params = default_params
                    validation_result["parameter_count"] = len(params)
                    if constraints:
                        if isinstance(constraints, str):
                            constraint_dict = json.loads(constraints)
                        else:
                            constraint_dict = constraints
                        for param_name, param_value in params.items():
                            if param_name in constraint_dict:
                                constraint = constraint_dict[param_name]
                                if "type" in constraint:
                                    expected_type = constraint["type"]
                                    if expected_type == "boolean" and (
                                        not isinstance(param_value, bool)
                                    ):
                                        validation_result[
                                            "constraint_violations"
                                        ].append(
                                            f"{param_name}: expected boolean, got {type(param_value).__name__}"
                                        )
                                    elif expected_type == "number" and (
                                        not isinstance(param_value, (int, float))
                                    ):
                                        validation_result[
                                            "constraint_violations"
                                        ].append(
                                            f"{param_name}: expected number, got {type(param_value).__name__}"
                                        )
                                if "min" in constraint and isinstance(
                                    param_value, (int, float)
                                ):
                                    if param_value < constraint["min"]:
                                        validation_result[
                                            "constraint_violations"
                                        ].append(
                                            f"{param_name}: {param_value} < minimum {constraint['min']}"
                                        )
                                if "max" in constraint and isinstance(
                                    param_value, (int, float)
                                ):
                                    if param_value > constraint["max"]:
                                        validation_result[
                                            "constraint_violations"
                                        ].append(
                                            f"{param_name}: {param_value} > maximum {constraint['max']}"
                                        )
                if validation_result["constraint_violations"]:
                    validation_result["status"] = "constraint_violations"
                    validation_result["issues"].append(
                        "Parameter constraint violations found"
                    )
            except (json.JSONDecodeError, TypeError) as e:
                validation_result["status"] = "error"
                validation_result["issues"].append(f"Parameter parsing error: {e}")
            parameter_validation[rule_id] = validation_result
        return parameter_validation

    async def _analyze_rule_performance(
        self, db_session: AsyncSession
    ) -> dict[str, Any]:
        """Analyze historical rule performance data."""
        performance_analysis = {
            "rules_with_performance": 0,
            "total_performance_records": 0,
            "average_improvement_score": 0.0,
            "rule_performance": {},
        }
        try:
            result = await db_session.execute(
                text(
                    "\n                    SELECT\n                        rule_id,\n                        COUNT(*) as record_count,\n                        AVG(improvement_score) as avg_improvement,\n                        MAX(improvement_score) as max_improvement,\n                        MIN(improvement_score) as min_improvement\n                    FROM rule_performance\n                    GROUP BY rule_id\n                "
                )
            )
            total_records = 0
            total_improvement = 0.0
            for row in result.fetchall():
                rule_id, count, avg_improvement, max_improvement, min_improvement = row
                performance_analysis["rule_performance"][rule_id] = {
                    "performance_records": count,
                    "average_improvement": float(avg_improvement)
                    if avg_improvement
                    else 0.0,
                    "max_improvement": float(max_improvement)
                    if max_improvement
                    else 0.0,
                    "min_improvement": float(min_improvement)
                    if min_improvement
                    else 0.0,
                    "status": "good"
                    if avg_improvement and avg_improvement > 0.5
                    else "needs_improvement",
                }
                total_records += count
                total_improvement += float(avg_improvement) if avg_improvement else 0.0
            performance_analysis["rules_with_performance"] = len(
                performance_analysis["rule_performance"]
            )
            performance_analysis["total_performance_records"] = total_records
            performance_analysis["average_improvement_score"] = (
                total_improvement / len(performance_analysis["rule_performance"])
                if performance_analysis["rule_performance"]
                else 0.0
            )
        except Exception as e:
            self.logger.exception(f"Performance analysis failed: {e}")
            performance_analysis["error"] = str(e)
        return performance_analysis

    async def _generate_rule_recommendations(
        self, existence: dict, metadata: dict, parameters: dict, performance: dict
    ) -> list[str]:
        """Generate intelligent recommendations based on validation results."""
        recommendations = []
        missing_rules = [
            rule_id for rule_id, data in existence["rules"].items() if not data["found"]
        ]
        if missing_rules:
            recommendations.append(
                f"Missing critical rules: {', '.join(missing_rules)}. Run database migration to seed rules."
            )
        disabled_rules = [
            rule_id
            for rule_id, data in existence["rules"].items()
            if data["found"] and "Rule is disabled" in data.get("issues", [])
        ]
        if disabled_rules:
            recommendations.append(
                f"Enable disabled rules: {', '.join(disabled_rules)}"
            )
        if metadata["total_issues"] > 0:
            recommendations.append(
                f"Fix {metadata['total_issues']} metadata issues for optimal rule performance"
            )
        violation_rules = [
            rule_id
            for rule_id, data in parameters.items()
            if data["status"] == "constraint_violations"
        ]
        if violation_rules:
            recommendations.append(
                f"Fix parameter constraints for rules: {', '.join(violation_rules)}"
            )
        if performance.get("rules_with_performance", 0) == 0:
            recommendations.append(
                "No performance data available. Consider running training to collect rule effectiveness metrics."
            )
        else:
            poor_performers = [
                rule_id
                for rule_id, data in performance["rule_performance"].items()
                if data["status"] == "needs_improvement"
            ]
            if poor_performers:
                recommendations.append(
                    f"Rules with poor performance may need parameter tuning: {', '.join(poor_performers)}"
                )
        if existence["total_rules"] < existence["expected_rules"]:
            recommendations.append(
                "Consider running the complete rule seeding migration to ensure all rules are available"
            )
        return recommendations
