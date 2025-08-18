"""Apriori Application Service

Orchestrates Apriori analysis workflows and provides business logic interface for API endpoints.
Decouples the API layer from direct ML imports following clean architecture principles.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from prompt_improver.application.protocols.application_service_protocols import (
    ApplicationServiceProtocol,
)
# str moved to unified cache facade - using string keys
from prompt_improver.repositories.protocols.session_manager_protocol import (
    SessionManagerProtocol,
)
# Removed direct ML analyzer import - handled through repository
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
    AssociationRuleFilter,
)
from prompt_improver.core.domain.types import (
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
)
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    MLRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class AprioriApplicationServiceProtocol(ApplicationServiceProtocol):
    """Protocol for Apriori analysis application service."""

    async def execute_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        session_id: str | None = None,
    ) -> AprioriAnalysisResponseData:
        """Execute complete Apriori analysis workflow."""
        ...

    async def get_association_rules(
        self,
        filters: AssociationRuleFilter | None = None,
        sort_by: str = "lift",
        sort_desc: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve association rules with filtering and sorting."""
        ...

    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """Get patterns relevant to specific context items."""
        ...


class AprioriApplicationService:
    """
    Application service for Apriori association rule mining workflows.
    
    Orchestrates:
    - Apriori analysis configuration and execution
    - Association rule discovery and filtering
    - Pattern contextualization and insights
    - Caching of analysis results for performance
    - Business logic validation and error handling
    """

    def __init__(
        self,
        session_manager: SessionManagerProtocol,
        apriori_repository: AprioriRepositoryProtocol,
        ml_repository: MLRepositoryProtocol,
        cache_manager,  # Cache manager from performance layer
    ):
        """
        Initialize the Apriori application service.
        
        Args:
            session_manager: Session manager protocol for transaction management
            apriori_repository: Repository for Apriori-specific data operations
            ml_repository: ML repository for feature data access
            cache_manager: Cache manager for performance optimization
        """
        self.session_manager = session_manager
        self.apriori_repository = apriori_repository
        self.ml_repository = ml_repository
        self.cache_manager = cache_manager
        self.logger = logger
        # Analyzer handling moved to repository layer

    async def initialize(self) -> None:
        """Initialize the Apriori application service."""
        self.logger.info("Initializing AprioriApplicationService")

    async def cleanup(self) -> None:
        """Clean up Apriori application service resources."""
        self.logger.info("Cleaning up AprioriApplicationService")

    # Removed _get_apriori_analyzer - analysis is now handled through repository

    async def execute_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        session_id: str | None = None,
    ) -> AprioriAnalysisResponseData:
        """
        Execute complete Apriori analysis workflow.
        
        Orchestrates:
        1. Configuration validation
        2. Data extraction and preprocessing 
        3. Apriori algorithm execution
        4. Rule generation and scoring
        5. Results storage and caching
        6. Business insights generation
        
        Args:
            request: Apriori analysis configuration
            session_id: Optional session identifier for tracking
            
        Returns:
            AprioriAnalysisResponseData with analysis results
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting Apriori analysis {analysis_id}")
            
            # Check cache first for identical requests
            cache_key = self._create_analysis_cache_key(request)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                self.logger.info(f"Returning cached Apriori analysis result")
                return cached_result
            
            # Validate configuration
            validation_result = await self._validate_analysis_config(request)
            if not validation_result["valid"]:
                return AprioriAnalysisResponseData(
                    status="error",
                    error=validation_result["error"],
                    analysis_run_id=analysis_id,
                )
            
            # Execute analysis within transaction boundary
            async with self.session_manager.get_session() as db_session:
                try:
                    # Get Apriori analyzer with proper dependencies
                    analyzer = await self._get_apriori_analyzer()
                    
                    # Create Apriori configuration
                    apriori_config = AprioriConfig(
                        min_support=request.min_support,
                        min_confidence=request.min_confidence,
                        min_lift=request.min_lift,
                        max_itemset_length=request.max_itemset_length,
                        transaction_window=getattr(request, 'transaction_window', 1000),
                    )
                    analyzer.config = apriori_config
                    
                    # Execute Apriori analysis
                    analysis_result = await analyzer.run_apriori_analysis(
                        window_days=request.window_days,
                        min_support=request.min_support,
                        min_confidence=request.min_confidence,
                    )
                    
                    if analysis_result.get("status") == "error":
                        return AprioriAnalysisResponseData(
                            status="error",
                            error=analysis_result.get("error", "Analysis failed"),
                            analysis_run_id=analysis_id,
                        )
                    
                    # Store analysis results using repository
                    result_response = await self.apriori_repository.run_apriori_analysis(request)
                    
                    # Store in cache for future requests
                    await self.cache_manager.set(
                        cache_key,
                        result_response,
                        ttl_seconds=3600,  # 1 hour cache
                    )
                    
                    await db_session.commit()
                    
                    end_time = datetime.now(timezone.utc)
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    self.logger.info(
                        f"Apriori analysis {analysis_id} completed in {duration_seconds:.2f}s"
                    )
                    
                    return result_response
                    
                except Exception as e:
                    await db_session.rollback()
                    self.logger.error(f"Error in Apriori analysis transaction: {e}")
                    return AprioriAnalysisResponseData(
                        status="error",
                        error=f"Analysis transaction failed: {str(e)}",
                        analysis_run_id=analysis_id,
                    )
                    
        except Exception as e:
            self.logger.error(f"Error in Apriori analysis workflow: {e}")
            return AprioriAnalysisResponseData(
                status="error",
                error=f"Workflow execution failed: {str(e)}",
                analysis_run_id=analysis_id,
            )

    async def get_association_rules(
        self,
        filters: AssociationRuleFilter | None = None,
        sort_by: str = "lift",
        sort_desc: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve association rules with filtering and sorting.
        
        Args:
            filters: Optional filters for rule selection
            sort_by: Field to sort by (lift, confidence, support)
            sort_desc: Sort in descending order
            limit: Maximum number of rules to return
            
        Returns:
            List of association rules with metrics
        """
        try:
            # Check cache first
            cache_key = self._create_rules_cache_key(filters, sort_by, sort_desc, limit)
            cached_rules = await self.cache_manager.get(cache_key)
            if cached_rules is not None:
                return cached_rules
            
            # Retrieve rules using repository
            rules = await self.apriori_repository.get_association_rules(
                filters=filters,
                sort_by=sort_by,
                sort_desc=sort_desc,
                limit=limit,
            )
            
            # Convert to business-friendly format
            rule_list = []
            for rule in rules:
                rule_dict = {
                    "id": rule.id,
                    "antecedents": rule.antecedents,
                    "consequents": rule.consequents,
                    "support": rule.support,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                    "conviction": rule.conviction,
                    "rule_strength": rule.rule_strength,
                    "business_insight": rule.business_insight,
                    "pattern_category": rule.pattern_category,
                    "created_at": rule.created_at.isoformat(),
                    "discovery_run_id": rule.discovery_run_id,
                }
                rule_list.append(rule_dict)
            
            # Cache the results
            await self.cache_manager.set(
                cache_key,
                rule_list,
                ttl_seconds=1800,  # 30 minute cache
            )
            
            self.logger.info(f"Retrieved {len(rule_list)} association rules")
            return rule_list
            
        except Exception as e:
            self.logger.error(f"Error retrieving association rules: {e}")
            raise

    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Get patterns relevant to specific context items.
        
        Args:
            context_items: List of items representing current context
            min_confidence: Minimum confidence threshold for patterns
            
        Returns:
            Dictionary with contextualized patterns and recommendations
        """
        try:
            # Check cache first
            cache_key = self._create_context_cache_key(context_items, min_confidence)
            cached_patterns = await self.cache_manager.get(cache_key)
            if cached_patterns is not None:
                return cached_patterns
            
            async with self.session_manager.get_session() as db_session:
                # Execute contextualized analysis through repository
                
                # Use repository to get context-relevant rules
                context_filters = AssociationRuleFilter(
                    min_confidence=min_confidence,
                    pattern_category=None,  # Include all categories for context analysis
                )
                
                all_rules = await self.apriori_repository.get_association_rules(
                    filters=context_filters,
                    sort_by="confidence",
                    sort_desc=True,
                    limit=200,  # Get more rules for context analysis
                )
                
                # Filter rules relevant to context items
                relevant_rules = []
                for rule in all_rules:
                    # Check if any context items appear in antecedents or consequents
                    rule_items = set(rule.antecedents + rule.consequents)
                    if any(item in rule_items for item in context_items):
                        relevant_rules.append({
                            "antecedents": rule.antecedents,
                            "consequents": rule.consequents,
                            "confidence": rule.confidence,
                            "lift": rule.lift,
                            "business_insight": rule.business_insight,
                            "pattern_category": rule.pattern_category,
                        })
                
                # Generate contextualized recommendations
                recommendations = await self._generate_context_recommendations(
                    context_items, relevant_rules
                )
                
                result = {
                    "context_items": context_items,
                    "relevant_patterns_count": len(relevant_rules),
                    "relevant_patterns": relevant_rules[:20],  # Top 20 most relevant
                    "recommendations": recommendations,
                    "confidence_threshold": min_confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                # Cache the results
                await self.cache_manager.set(
                    cache_key,
                    result,
                    ttl_seconds=900,  # 15 minute cache for context patterns
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting contextualized patterns: {e}")
            raise

    async def _validate_analysis_config(self, request: AprioriAnalysisRequestData) -> Dict[str, Any]:
        """Validate Apriori analysis configuration."""
        if request.min_support <= 0 or request.min_support > 1:
            return {
                "valid": False,
                "error": "min_support must be between 0 and 1"
            }
        
        if request.min_confidence <= 0 or request.min_confidence > 1:
            return {
                "valid": False,
                "error": "min_confidence must be between 0 and 1"
            }
        
        if request.min_lift < 0:
            return {
                "valid": False,
                "error": "min_lift must be non-negative"
            }
        
        return {"valid": True}

    async def _generate_context_recommendations(
        self,
        context_items: List[str],
        relevant_rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on context and relevant rules."""
        recommendations = []
        
        # Group rules by consequents to find most recommended actions
        consequent_scores = {}
        for rule in relevant_rules:
            for consequent in rule["consequents"]:
                if consequent not in context_items:  # Don't recommend what's already present
                    score = rule["confidence"] * rule["lift"]  # Combined score
                    if consequent in consequent_scores:
                        consequent_scores[consequent] = max(consequent_scores[consequent], score)
                    else:
                        consequent_scores[consequent] = score
        
        # Sort and create recommendations
        sorted_consequents = sorted(
            consequent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10 recommendations
        
        for consequent, score in sorted_consequents:
            # Find the best rule supporting this recommendation
            best_rule = max(
                [r for r in relevant_rules if consequent in r["consequents"]],
                key=lambda r: r["confidence"] * r["lift"]
            )
            
            recommendations.append({
                "recommended_action": consequent,
                "confidence": best_rule["confidence"],
                "lift": best_rule["lift"],
                "supporting_rule": {
                    "antecedents": best_rule["antecedents"],
                    "consequents": best_rule["consequents"],
                },
                "business_insight": best_rule["business_insight"],
                "recommendation_score": score,
            })
        
        return recommendations

    def _create_analysis_cache_key(self, request: AprioriAnalysisRequestData) -> str:
        """Create cache key for Apriori analysis request."""
        return f"apriori_analysis:{request.min_support}:{request.min_confidence}:{request.min_lift}:{request.window_days}"

    def _create_rules_cache_key(
        self,
        filters: AssociationRuleFilter | None,
        sort_by: str,
        sort_desc: bool,
        limit: int,
    ) -> str:
        """Create cache key for association rules request."""
        filter_key = "none"
        if filters:
            filter_key = f"{filters.min_confidence}:{filters.min_lift}:{filters.pattern_category}"
        
        return f"association_rules:{filter_key}:{sort_by}:{sort_desc}:{limit}"

    def _create_context_cache_key(
        self,
        context_items: List[str],
        min_confidence: float,
    ) -> str:
        """Create cache key for contextualized patterns request."""
        context_key = ":".join(sorted(context_items))
        return f"context_patterns:{context_key}:{min_confidence}"