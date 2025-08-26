"""Apriori Algorithm API Endpoints

This module provides REST API endpoints for Apriori association rule mining
and pattern discovery, integrating with the ML pipeline for comprehensive
prompt improvement insights.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from prompt_improver.database.composition import DatabaseServices
from prompt_improver.database import get_database_services, create_database_services, ManagerMode
from prompt_improver.shared.interfaces.protocols.application import (
    AprioriApplicationServiceProtocol,
    PatternApplicationServiceProtocol,
)
from prompt_improver.application.services.apriori_application_service import (
    AprioriApplicationService,
)
from prompt_improver.application.services.pattern_application_service import (
    PatternApplicationService,
)
# Cache manager handled through dependency injection
from prompt_improver.repositories.factory import get_apriori_repository
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
)

logger = logging.getLogger(__name__)


# Clean dependency function for ML training services
async def get_ml_database_services() -> DatabaseServices:
    """Get database services for ML training operations."""
    services = await get_database_services(ManagerMode.ML_TRAINING)
    if services is None:
        services = await create_database_services(ManagerMode.ML_TRAINING)
    return services


apriori_router = APIRouter(
    prefix="/api/v1/apriori", tags=["apriori", "pattern-discovery"]
)


async def get_apriori_repository_dep(
    db_manager: DatabaseServices = Depends(get_ml_database_services),
) -> AprioriRepositoryProtocol:
    """Get apriori repository with database services"""
    return await get_apriori_repository(db_manager)


async def get_apriori_application_service(
    db_manager: DatabaseServices = Depends(get_ml_database_services),
    apriori_repository: AprioriRepositoryProtocol = Depends(get_apriori_repository_dep),
) -> AprioriApplicationServiceProtocol:
    """Get Apriori application service with proper dependencies"""
    from prompt_improver.repositories.factory import get_ml_repository
    from prompt_improver.core.caching import create_cache_manager
    
    ml_repository = await get_ml_repository(db_manager)
    cache_manager = await create_cache_manager()
    
    service = AprioriApplicationService(
        db_services=db_manager,
        apriori_repository=apriori_repository,
        ml_repository=ml_repository,
        cache_manager=cache_manager,
    )
    await service.initialize()
    return service


async def get_pattern_application_service(
    db_manager: DatabaseServices = Depends(get_ml_database_services),
    apriori_repository: AprioriRepositoryProtocol = Depends(get_apriori_repository_dep),
) -> PatternApplicationServiceProtocol:
    """Get Pattern application service with proper dependencies"""
    from prompt_improver.repositories.factory import get_ml_repository
    from prompt_improver.core.caching import create_cache_manager
    
    ml_repository = await get_ml_repository(db_manager)
    cache_manager = await create_cache_manager()
    
    service = PatternApplicationService(
        db_services=db_manager,
        apriori_repository=apriori_repository,
        ml_repository=ml_repository,
        cache_manager=cache_manager,
    )
    await service.initialize()
    return service


@apriori_router.post("/analyze", response_model=AprioriAnalysisResponse)
async def run_apriori_analysis(
    request: AprioriAnalysisRequest,
    apriori_service: AprioriApplicationServiceProtocol = Depends(get_apriori_application_service),
) -> AprioriAnalysisResponse:
    """Run Apriori association rule mining analysis.

    This endpoint performs comprehensive Apriori analysis on prompt improvement data
    to discover association rules between prompt characteristics, rule applications,
    and improvement outcomes.

    Args:
        request: Configuration parameters for Apriori analysis
        apriori_analyzer: AprioriAnalyzer service instance
        db_manager: Database manager

    Returns:
        AprioriAnalysisResponse with discovered patterns and insights

    Raises:
        HTTPException: If analysis fails or insufficient data
    """
    try:
        logger.info(f"Starting Apriori analysis with config: {request.model_dump()}")

        # Use application service for orchestrated analysis
        result = await apriori_service.execute_apriori_analysis(request)
        
        if result.status == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Apriori analysis failed: {result.error}",
            )
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apriori analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during Apriori analysis: {e!s}",
        )


@apriori_router.post("/discover-patterns", response_model=PatternDiscoveryResponse)
async def comprehensive_pattern_discovery(
    request: PatternDiscoveryRequest,
    pattern_service: PatternApplicationServiceProtocol = Depends(get_pattern_application_service),
) -> PatternDiscoveryResponse:
    """Run comprehensive pattern discovery combining traditional ML with Apriori analysis.

    This endpoint performs advanced pattern discovery using multiple algorithms:
    - Traditional ML parameter analysis
    - HDBSCAN clustering for density-based patterns
    - FP-Growth for frequent pattern mining
    - Apriori for association rule mining
    - Semantic analysis for rule relationships

    Args:
        request: Configuration for pattern discovery
        ml_service: ML service instance
        db_manager: Database manager

    Returns:
        PatternDiscoveryResponse with comprehensive pattern analysis

    Raises:
        HTTPException: If pattern discovery fails
    """
    try:
        logger.info(
            f"Starting comprehensive pattern discovery: {request.model_dump()}"
        )
        
        # Use application service for orchestrated pattern discovery
        result = await pattern_service.execute_comprehensive_pattern_discovery(request)
        
        if result.status == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pattern discovery failed: {result.business_insights.get('error', 'Unknown error')}",
            )
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive pattern discovery failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during pattern discovery: {e!s}",
        )


@apriori_router.get("/rules", response_model=list[dict[str, Any]])
async def get_association_rules(
    apriori_service: AprioriApplicationServiceProtocol = Depends(get_apriori_application_service),
    min_confidence: float = 0.6,
    min_lift: float = 1.0,
    pattern_category: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve stored association rules with filtering options.

    Args:
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        pattern_category: Filter by pattern category
        limit: Maximum number of rules to return
        db_manager: Database manager

    Returns:
        List of association rules with metrics and insights
    """
    try:
        from prompt_improver.repositories.protocols.apriori_repository_protocol import AssociationRuleFilter

        # Create filter from parameters
        filters = AssociationRuleFilter(
            min_confidence=min_confidence,
            min_lift=min_lift,
            pattern_category=pattern_category,
        )

        # Use application service to get rules
        rule_list = await apriori_service.get_association_rules(
            filters=filters,
            sort_by="lift",
            sort_desc=True,
            limit=limit,
        )

        logger.info(f"Retrieved {len(rule_list)} association rules")
        return rule_list
    except Exception as e:
        logger.error(f"Error retrieving association rules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving association rules: {e!s}",
        )


@apriori_router.post("/contextualized-patterns")
async def get_contextualized_patterns(
    context_items: list[str],
    apriori_service: AprioriApplicationServiceProtocol = Depends(get_apriori_application_service),
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Get patterns relevant to a specific context using Apriori and ML analysis.

    This endpoint finds patterns relevant to the current prompt improvement context
    by analyzing association rules and traditional ML patterns.

    Args:
        context_items: Items representing current context (rules, characteristics)
        min_confidence: Minimum confidence for returned patterns
        ml_service: ML service instance
        db_manager: Database manager

    Returns:
        Dictionary with contextualized patterns and recommendations
    """
    try:
        logger.info(f"Getting contextualized patterns for: {context_items}")
        
        # Use application service for contextualized pattern analysis
        results = await apriori_service.get_contextualized_patterns(
            context_items=context_items,
            min_confidence=min_confidence,
        )
        
        return results
    except Exception as e:
        logger.error(f"Error getting contextualized patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing contextualized patterns: {e!s}",
        )


@apriori_router.get("/discovery-runs", response_model=list[dict[str, Any]])
async def get_discovery_runs(
    pattern_service: PatternApplicationServiceProtocol = Depends(get_pattern_application_service),
    limit: int = 20,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve historical pattern discovery runs.

    Args:
        limit: Maximum number of runs to return
        status_filter: Filter by run status (running, completed, failed)
        db_manager: Database manager

    Returns:
        List of discovery run metadata
    """
    try:
        from prompt_improver.repositories.protocols.apriori_repository_protocol import PatternDiscoveryFilter
        from prompt_improver.ml.types import PatternDiscoveryResponse

        # Create filter from parameters
        filters = (
            PatternDiscoveryFilter(status=status_filter) if status_filter else None
        )

        # Use application service to get discovery runs
        run_list = await pattern_service.get_pattern_discoveries(
            filters=filters,
            sort_by="created_at",
            sort_desc=True,
            limit=limit,
        )
        
        return run_list
    except Exception as e:
        logger.error(f"Error retrieving discovery runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving discovery runs: {e!s}",
        )


@apriori_router.get("/insights/{discovery_run_id}")
async def get_discovery_insights(
    discovery_run_id: str,
    pattern_service: PatternApplicationServiceProtocol = Depends(get_pattern_application_service),
) -> dict[str, Any]:
    """Get detailed insights for a specific discovery run.

    Args:
        discovery_run_id: ID of the discovery run
        db_manager: Database manager

    Returns:
        Detailed insights and patterns from the discovery run
    """
    try:
        # Use application service to get comprehensive discovery results summary
        insights = await pattern_service.get_discovery_insights(discovery_run_id)
        
        return insights
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error retrieving discovery insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving insights: {e!s}",
        )


# Helper functions removed - now handled by application services
