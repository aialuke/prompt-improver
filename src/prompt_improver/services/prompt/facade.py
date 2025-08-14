"""PromptServiceFacade - Unified interface for prompt improvement operations.

This facade coordinates between three focused services:
1. PromptAnalysisService - Prompt analysis and improvement logic
2. RuleApplicationService - Rule execution and validation  
3. ValidationService - Input validation and business rule checking

Replaces the 1,544-line prompt_improvement.py god object while maintaining
identical public API. Follows Clean Architecture principles with protocol-based DI.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import UUID

from prompt_improver.core.protocols.prompt_service.prompt_protocols import (
    PromptServiceFacadeProtocol,
    PromptAnalysisServiceProtocol,
    RuleApplicationServiceProtocol,
    ValidationServiceProtocol,
)
from prompt_improver.database.models import UserFeedback
from prompt_improver.rule_engine.base import BasePromptRule
from prompt_improver.utils.datetime_utils import aware_utc_now

from .prompt_analysis_service import PromptAnalysisService
from .rule_application_service import RuleApplicationService
from .validation_service import ValidationService

logger = logging.getLogger(__name__)


class PromptServiceFacade(PromptServiceFacadeProtocol):
    """
    Unified PromptServiceFacade implementing 2025 best practices.
    
    This facade provides a single entry point for all prompt improvement operations,
    coordinating between specialized services while maintaining high performance
    and reliability.
    
    Key Features:
    - Unified interface for all prompt improvement operations
    - Intelligent service routing and coordination
    - Advanced validation and business rule enforcement
    - Performance monitoring and health checking
    - Graceful degradation under load
    - Protocol-based dependency injection
    """

    def __init__(
        self,
        analysis_service: Optional[PromptAnalysisServiceProtocol] = None,
        rule_application_service: Optional[RuleApplicationServiceProtocol] = None,
        validation_service: Optional[ValidationServiceProtocol] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize PromptServiceFacade with dependency injection."""
        self.config = config or {}
        self.logger = logger
        
        # Initialize services with dependency injection
        self.analysis_service = analysis_service or PromptAnalysisService()
        self.rule_application_service = rule_application_service or RuleApplicationService()
        self.validation_service = validation_service or ValidationService()
        
        # Performance monitoring
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        self._last_health_check = datetime.now()
        
        # Circuit breaker state for services
        self._service_health = {
            "analysis": True,
            "rule_application": True,
            "validation": True
        }
        
        # Configuration
        self._performance_targets = {
            "max_response_time_ms": 5000,
            "max_validation_time_ms": 1000,
            "max_analysis_time_ms": 3000
        }

    async def improve_prompt(
        self,
        prompt: str,
        user_id: Optional[UUID] = None,
        session_id: Optional[UUID] = None,
        rules: Optional[List[BasePromptRule]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main method to improve a prompt."""
        start_time = datetime.now()
        
        try:
            self._request_count += 1
            
            # Create session ID if not provided
            if session_id is None:
                session_id = UUID(int=self._request_count)

            result = {
                "session_id": str(session_id),
                "user_id": str(user_id) if user_id else None,
                "original_prompt": prompt,
                "timestamp": aware_utc_now().isoformat(),
                "processing_steps": [],
                "performance_metrics": {}
            }

            # Step 1: Validate input
            validation_start = datetime.now()
            try:
                if self._service_health["validation"]:
                    validation_result = await self.validation_service.validate_prompt_input(
                        prompt, config.get("validation_constraints") if config else None
                    )
                    
                    if not validation_result["valid"]:
                        result["status"] = "validation_failed"
                        result["validation_result"] = validation_result
                        result["error"] = "Prompt validation failed"
                        return result
                    
                    result["validation_result"] = validation_result
                    result["processing_steps"].append({
                        "step": "validation",
                        "status": "completed",
                        "duration_ms": (datetime.now() - validation_start).total_seconds() * 1000
                    })
                else:
                    logger.warning("Validation service unavailable, skipping validation")
                    result["processing_steps"].append({
                        "step": "validation",
                        "status": "skipped",
                        "reason": "service_unavailable"
                    })
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                self._service_health["validation"] = False
                result["processing_steps"].append({
                    "step": "validation",
                    "status": "failed",
                    "error": str(e)
                })

            # Step 2: Sanitize prompt if needed
            sanitized_prompt = prompt
            if config and config.get("sanitize", False):
                try:
                    sanitization_level = config.get("sanitization_level", "standard")
                    sanitized_prompt = await self.validation_service.sanitize_prompt_content(
                        prompt, sanitization_level
                    )
                    result["sanitized"] = sanitized_prompt != prompt
                    result["sanitized_prompt"] = sanitized_prompt if result["sanitized"] else None
                except Exception as e:
                    logger.warning(f"Sanitization failed, using original prompt: {e}")

            # Step 3: Analyze prompt
            analysis_start = datetime.now()
            try:
                if self._service_health["analysis"]:
                    # Create a temporary prompt ID for analysis
                    temp_prompt_id = UUID(int=hash(prompt) % (10**16))
                    
                    analysis_result = await self.analysis_service.analyze_prompt(
                        temp_prompt_id, session_id, config
                    )
                    result["analysis"] = analysis_result
                    result["processing_steps"].append({
                        "step": "analysis",
                        "status": "completed",
                        "duration_ms": (datetime.now() - analysis_start).total_seconds() * 1000
                    })
                else:
                    logger.warning("Analysis service unavailable, skipping analysis")
                    result["processing_steps"].append({
                        "step": "analysis",
                        "status": "skipped",
                        "reason": "service_unavailable"
                    })
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self._service_health["analysis"] = False
                result["processing_steps"].append({
                    "step": "analysis",
                    "status": "failed",
                    "error": str(e)
                })

            # Step 4: Apply rules
            rule_application_start = datetime.now()
            try:
                if self._service_health["rule_application"]:
                    # Use provided rules or get default rules
                    rules_to_apply = rules or await self._get_default_rules()
                    
                    if rules_to_apply:
                        # Validate rule compatibility
                        compatibility_result = await self.rule_application_service.validate_rule_compatibility(
                            rules_to_apply
                        )
                        
                        if compatibility_result["overall_compatible"]:
                            # Apply rules
                            rule_application_result = await self.rule_application_service.apply_rules(
                                sanitized_prompt, rules_to_apply, session_id, config
                            )
                            result["rule_application"] = rule_application_result
                            result["improved_prompt"] = rule_application_result["final_prompt"]
                        else:
                            result["rule_application"] = {
                                "status": "compatibility_issues",
                                "compatibility_result": compatibility_result,
                                "final_prompt": sanitized_prompt
                            }
                            result["improved_prompt"] = sanitized_prompt
                    else:
                        result["improved_prompt"] = sanitized_prompt
                        result["rule_application"] = {
                            "status": "no_rules_available",
                            "final_prompt": sanitized_prompt
                        }
                    
                    result["processing_steps"].append({
                        "step": "rule_application",
                        "status": "completed",
                        "duration_ms": (datetime.now() - rule_application_start).total_seconds() * 1000
                    })
                else:
                    logger.warning("Rule application service unavailable, skipping rule application")
                    result["improved_prompt"] = sanitized_prompt
                    result["processing_steps"].append({
                        "step": "rule_application",
                        "status": "skipped",
                        "reason": "service_unavailable"
                    })
            except Exception as e:
                logger.error(f"Rule application failed: {e}")
                self._service_health["rule_application"] = False
                result["improved_prompt"] = sanitized_prompt
                result["processing_steps"].append({
                    "step": "rule_application",
                    "status": "failed",
                    "error": str(e)
                })

            # Step 5: Generate quality assessment
            if "improved_prompt" in result and result["improved_prompt"] != prompt:
                try:
                    quality_assessment = await self.analysis_service.evaluate_improvement_quality(
                        prompt, result["improved_prompt"]
                    )
                    result["quality_assessment"] = quality_assessment
                except Exception as e:
                    logger.warning(f"Quality assessment failed: {e}")

            # Step 6: Get ML recommendations if enabled
            if config and config.get("enable_ml_recommendations", False):
                try:
                    ml_recommendations = await self.analysis_service.get_ml_recommendations(
                        prompt, session_id
                    )
                    result["ml_recommendations"] = ml_recommendations
                except Exception as e:
                    logger.warning(f"ML recommendations failed: {e}")

            # Calculate performance metrics
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            self._total_response_time += total_time
            
            result["performance_metrics"] = {
                "total_processing_time_ms": total_time,
                "average_response_time_ms": self._total_response_time / self._request_count,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "success_rate": (self._request_count - self._error_count) / self._request_count
            }

            result["status"] = "success"
            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error improving prompt: {e}")
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            return {
                "status": "error",
                "error": str(e),
                "session_id": str(session_id) if session_id else None,
                "original_prompt": prompt,
                "timestamp": aware_utc_now().isoformat(),
                "performance_metrics": {
                    "total_processing_time_ms": total_time,
                    "error_count": self._error_count
                }
            }

    async def get_session_summary(
        self,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Get summary of an improvement session."""
        try:
            # This would typically fetch session data from repository
            # For now, return a summary based on session ID
            summary = {
                "session_id": str(session_id),
                "status": "active",
                "created_at": aware_utc_now().isoformat(),
                "operations_count": 1,
                "total_processing_time_ms": 0,
                "success_rate": 1.0,
                "last_activity": aware_utc_now().isoformat()
            }
            
            return summary

        except Exception as e:
            logger.error(f"Error getting session summary for {session_id}: {e}")
            raise

    async def process_feedback(
        self,
        feedback: UserFeedback,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Process user feedback for a session."""
        try:
            # Process feedback and potentially trigger optimization
            feedback_result = {
                "feedback_id": getattr(feedback, 'id', None),
                "session_id": str(session_id),
                "processed_at": aware_utc_now().isoformat(),
                "status": "processed"
            }
            
            # Check if feedback should trigger optimization
            if hasattr(feedback, 'rating') and feedback.rating:
                if feedback.rating < 3:  # Low rating triggers analysis
                    feedback_result["triggered_analysis"] = True
                    feedback_result["recommendation"] = "Consider reviewing prompt improvement rules"

            return feedback_result

        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all prompt services."""
        try:
            health_status = {
                "overall_health": "healthy",
                "timestamp": aware_utc_now().isoformat(),
                "services": {
                    "analysis_service": {
                        "healthy": self._service_health["analysis"],
                        "status": "operational" if self._service_health["analysis"] else "degraded"
                    },
                    "rule_application_service": {
                        "healthy": self._service_health["rule_application"],
                        "status": "operational" if self._service_health["rule_application"] else "degraded"
                    },
                    "validation_service": {
                        "healthy": self._service_health["validation"],
                        "status": "operational" if self._service_health["validation"] else "degraded"
                    }
                },
                "performance_metrics": {
                    "total_requests": self._request_count,
                    "error_count": self._error_count,
                    "success_rate": (self._request_count - self._error_count) / max(self._request_count, 1),
                    "average_response_time_ms": self._total_response_time / max(self._request_count, 1)
                }
            }

            # Determine overall health
            unhealthy_services = [
                service for service, healthy in self._service_health.items() 
                if not healthy
            ]
            
            if unhealthy_services:
                health_status["overall_health"] = "degraded"
                health_status["unhealthy_services"] = unhealthy_services

            return health_status

        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "overall_health": "error",
                "error": str(e),
                "timestamp": aware_utc_now().isoformat()
            }

    async def _get_default_rules(self) -> List[BasePromptRule]:
        """Get default rules for prompt improvement."""
        # This would typically load rules from repository
        # For now, return empty list as rules would be injected
        return []

    async def _check_business_rules(
        self,
        operation: str,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check business rules for operation."""
        try:
            return await self.validation_service.check_business_rules(
                operation, data, context
            )
        except Exception as e:
            logger.error(f"Business rule check failed for {operation}: {e}")
            return False  # Fail secure

    def _update_service_health(self, service: str, healthy: bool) -> None:
        """Update health status for a service."""
        if service in self._service_health:
            self._service_health[service] = healthy
            logger.info(f"Service {service} health updated to: {healthy}")

    def _should_circuit_break(self, service: str) -> bool:
        """Check if service should be circuit broken."""
        return not self._service_health.get(service, True)