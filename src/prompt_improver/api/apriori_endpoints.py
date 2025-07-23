"""Apriori Algorithm API Endpoints

This module provides REST API endpoints for Apriori association rule mining
and pattern discovery, integrating with the ML pipeline for comprehensive
prompt improvement insights.
"""

import logging
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import DBSession
from ..database.connection import DatabaseManager
from ..database.models import (
    AprioriAnalysisRequest,
    AprioriAnalysisResponse,
    AprioriAssociationRule,
    AprioriPatternDiscovery,
    PatternDiscoveryRequest,
    PatternDiscoveryResponse,
)
from ..ml.learning.patterns.advanced_pattern_discovery import AdvancedPatternDiscovery
from ..ml.learning.patterns.apriori_analyzer import AprioriAnalyzer, AprioriConfig
from ..ml.core.ml_integration import MLModelService, get_ml_service

logger = logging.getLogger(__name__)

apriori_router = APIRouter(
    prefix="/api/v1/apriori", tags=["apriori", "pattern-discovery"]
)


async def get_apriori_analyzer() -> AprioriAnalyzer:
    """Dependency to get AprioriAnalyzer instance with proper database configuration"""
    # Use secure database configuration from environment variables
    import os
    from ..database.config import get_database_config

    db_config = get_database_config()
    # Use the secure database URL from config
    database_url = db_config.database_url_sync

    db_manager = DatabaseManager(database_url)
    return AprioriAnalyzer(db_manager=db_manager)


async def get_pattern_discovery() -> AdvancedPatternDiscovery:
    """Dependency to get AdvancedPatternDiscovery instance with proper database configuration"""
    # Use secure database configuration from environment variables
    import os
    from ..database.config import get_database_config

    db_config = get_database_config()
    # Use the secure database URL from config
    database_url = db_config.database_url_sync

    db_manager = DatabaseManager(database_url)
    return AdvancedPatternDiscovery(db_manager=db_manager)


@apriori_router.post("/analyze", response_model=AprioriAnalysisResponse)
async def run_apriori_analysis(
    request: AprioriAnalysisRequest,
    db_session: DBSession,
    apriori_analyzer: AprioriAnalyzer = Depends(get_apriori_analyzer),
) -> AprioriAnalysisResponse:
    """Run Apriori association rule mining analysis.

    This endpoint performs comprehensive Apriori analysis on prompt improvement data
    to discover association rules between prompt characteristics, rule applications,
    and improvement outcomes.

    Args:
        request: Configuration parameters for Apriori analysis
        apriori_analyzer: AprioriAnalyzer service instance
        db_session: Database session

    Returns:
        AprioriAnalysisResponse with discovered patterns and insights

    Raises:
        HTTPException: If analysis fails or insufficient data
    """
    try:
        logger.info(f"Starting Apriori analysis with config: {request.dict()}")

        # Update analyzer configuration
        config = AprioriConfig(
            min_support=request.min_support,
            min_confidence=request.min_confidence,
            min_lift=request.min_lift,
            max_itemset_length=request.max_itemset_length,
            verbose=True,
        )
        apriori_analyzer.config = config

        # Run Apriori analysis
        results = apriori_analyzer.analyze_patterns(
            window_days=request.window_days, save_to_database=request.save_to_database
        )

        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Apriori analysis failed: {results['error']}",
            )

        # Generate discovery run ID for tracking
        discovery_run_id = str(uuid.uuid4())

        # Store analysis metadata in database if requested
        if request.save_to_database:
            await _store_apriori_analysis_metadata(
                db_session, discovery_run_id, request, results
            )

        return AprioriAnalysisResponse(
            discovery_run_id=discovery_run_id,
            transaction_count=results.get("transaction_count", 0),
            frequent_itemsets_count=results.get("frequent_itemsets_count", 0),
            association_rules_count=results.get("association_rules_count", 0),
            execution_time_seconds=0.0,  # Will be calculated from results
            top_itemsets=results.get("top_itemsets", []),
            top_rules=results.get("top_rules", []),
            pattern_insights=results.get("pattern_insights", {}),
            config=results.get("config", {}),
            status="success",
            timestamp=results.get("timestamp", datetime.now().isoformat()),
        )

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
    db_session: DBSession,
    ml_service: Annotated[MLModelService, Depends(get_ml_service)],
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
        db_session: Database session

    Returns:
        PatternDiscoveryResponse with comprehensive pattern analysis

    Raises:
        HTTPException: If pattern discovery fails
    """
    try:
        logger.info(f"Starting comprehensive pattern discovery: {request.dict()}")

        # Run enhanced pattern discovery
        results = await ml_service.discover_patterns(
            db_session=db_session,
            min_effectiveness=request.min_effectiveness,
            min_support=request.min_support,
            use_advanced_discovery=request.use_advanced_discovery,
            include_apriori=request.include_apriori,
        )

        if results.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pattern discovery failed: {results.get('error')}",
            )

        # Generate discovery run ID
        discovery_run_id = str(uuid.uuid4())

        # Store comprehensive results in database
        await _store_pattern_discovery_results(db_session, discovery_run_id, results)

        return PatternDiscoveryResponse(
            status=results.get("status", "success"),
            discovery_run_id=discovery_run_id,
            traditional_patterns=results.get("traditional_patterns"),
            advanced_patterns=results.get("advanced_patterns"),
            apriori_patterns=results.get("apriori_patterns"),
            cross_validation=results.get("cross_validation"),
            unified_recommendations=results.get("unified_recommendations", []),
            business_insights=results.get("business_insights", {}),
            discovery_metadata=results.get("discovery_metadata", {}),
        )

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
    db_session: DBSession,
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
        db_session: Database session

    Returns:
        List of association rules with metrics and insights
    """
    try:
        from sqlalchemy import and_, select

        # Build query with filters
        query = select(AprioriAssociationRule)

        conditions = [
            AprioriAssociationRule.confidence >= min_confidence,
            AprioriAssociationRule.lift >= min_lift,
        ]

        if pattern_category:
            conditions.append(
                AprioriAssociationRule.pattern_category == pattern_category
            )  # type: ignore[arg-type]

        query = query.where(and_(*conditions))  # type: ignore[arg-type]
        query = query.order_by(AprioriAssociationRule.rule_strength.desc())  # type: ignore[attr-defined]
        query = query.limit(limit)

        result = await db_session.execute(query)
        rules = result.scalars().all()

        # Convert to response format
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
    db_session: DBSession,
    ml_service: Annotated[MLModelService, Depends(get_ml_service)],
    min_confidence: float = 0.6,
) -> dict[str, Any]:
    """Get patterns relevant to a specific context using Apriori and ML analysis.

    This endpoint finds patterns relevant to the current prompt improvement context
    by analyzing association rules and traditional ML patterns.

    Args:
        context_items: Items representing current context (rules, characteristics)
        min_confidence: Minimum confidence for returned patterns
        ml_service: ML service instance
        db_session: Database session

    Returns:
        Dictionary with contextualized patterns and recommendations
    """
    try:
        logger.info(f"Getting contextualized patterns for: {context_items}")

        # Get contextualized patterns using ML service
        results = await ml_service.get_contextualized_patterns(
            context_items=context_items,
            db_session=db_session,
            min_confidence=min_confidence,
        )

        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Contextualized pattern analysis failed: {results['error']}",
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contextualized patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing contextualized patterns: {e!s}",
        )


@apriori_router.get("/discovery-runs", response_model=list[dict[str, Any]])
async def get_discovery_runs(
    db_session: DBSession, limit: int = 20, status_filter: str | None = None
) -> list[dict[str, Any]]:
    """Retrieve historical pattern discovery runs.

    Args:
        limit: Maximum number of runs to return
        status_filter: Filter by run status (running, completed, failed)
        db_session: Database session

    Returns:
        List of discovery run metadata
    """
    try:
        from sqlalchemy import select

        query = select(AprioriPatternDiscovery)

        if status_filter:
            query = query.where(AprioriPatternDiscovery.status == status_filter)  # type: ignore[arg-type]

        query = query.order_by(AprioriPatternDiscovery.created_at.desc())  # type: ignore[attr-defined]
        query = query.limit(limit)

        result = await db_session.execute(query)
        runs = result.scalars().all()

        run_list = []
        for run in runs:
            run_dict = {
                "discovery_run_id": run.discovery_run_id,
                "status": run.status,
                "transaction_count": run.transaction_count,
                "frequent_itemsets_count": run.frequent_itemsets_count,
                "association_rules_count": run.association_rules_count,
                "execution_time_seconds": run.execution_time_seconds,
                "created_at": run.created_at.isoformat(),
                "completed_at": run.completed_at.isoformat()
                if run.completed_at
                else None,
                "config": {
                    "min_support": run.min_support,
                    "min_confidence": run.min_confidence,
                    "min_lift": run.min_lift,
                    "data_window_days": run.data_window_days,
                },
            }
            run_list.append(run_dict)

        return run_list

    except Exception as e:
        logger.error(f"Error retrieving discovery runs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving discovery runs: {e!s}",
        )


@apriori_router.get("/insights/{discovery_run_id}")
async def get_discovery_insights(
    discovery_run_id: str, db_session: DBSession
) -> dict[str, Any]:
    """Get detailed insights for a specific discovery run.

    Args:
        discovery_run_id: ID of the discovery run
        db_session: Database session

    Returns:
        Detailed insights and patterns from the discovery run
    """
    try:
        from sqlalchemy import select

        # Get discovery run metadata
        query = select(AprioriPatternDiscovery).where(
            AprioriPatternDiscovery.discovery_run_id == discovery_run_id  # type: ignore[arg-type]
        )
        result = await db_session.execute(query)
        discovery_run = result.scalar_one_or_none()

        if not discovery_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Discovery run {discovery_run_id} not found",
            )

        # Get associated rules
        rules_query = (
            select(AprioriAssociationRule)
            .where(
                AprioriAssociationRule.discovery_run_id == discovery_run_id  # type: ignore[arg-type]
            )
            .order_by(AprioriAssociationRule.rule_strength.desc())
        )  # type: ignore[attr-defined]

        rules_result = await db_session.execute(rules_query)
        rules = rules_result.scalars().all()

        # Compile insights
        insights = {
            "discovery_run": {
                "id": discovery_run.discovery_run_id,
                "status": discovery_run.status,
                "execution_time_seconds": discovery_run.execution_time_seconds,
                "created_at": discovery_run.created_at.isoformat(),
                "config": {
                    "min_support": discovery_run.min_support,
                    "min_confidence": discovery_run.min_confidence,
                    "min_lift": discovery_run.min_lift,
                    "data_window_days": discovery_run.data_window_days,
                },
            },
            "summary": {
                "transaction_count": discovery_run.transaction_count,
                "frequent_itemsets_count": discovery_run.frequent_itemsets_count,
                "association_rules_count": discovery_run.association_rules_count,
                "top_patterns_summary": discovery_run.top_patterns_summary,
            },
            "top_rules": [
                {
                    "antecedents": rule.antecedents,
                    "consequents": rule.consequents,
                    "confidence": rule.confidence,
                    "lift": rule.lift,
                    "rule_strength": rule.rule_strength,
                    "business_insight": rule.business_insight,
                    "pattern_category": rule.pattern_category,
                }
                for rule in rules[:10]  # Top 10 rules
            ],
            "pattern_insights": discovery_run.pattern_insights,
            "quality_metrics": discovery_run.quality_metrics,
        }

        return insights

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving discovery insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving insights: {e!s}",
        )


# Helper functions for database operations


async def _store_apriori_analysis_metadata(
    db_session: DBSession,
    discovery_run_id: str,
    request: AprioriAnalysisRequest,
    results: dict[str, Any],
) -> None:
    """Store Apriori analysis metadata in database"""
    try:
        discovery_record = AprioriPatternDiscovery(  # type: ignore[call-arg]
            discovery_run_id=discovery_run_id,
            min_support=request.min_support,
            min_confidence=request.min_confidence,
            min_lift=request.min_lift,
            max_itemset_length=request.max_itemset_length,
            data_window_days=request.window_days,
            transaction_count=results.get("transaction_count", 0),
            frequent_itemsets_count=results.get("frequent_itemsets_count", 0),
            association_rules_count=results.get("association_rules_count", 0),
            top_patterns_summary=results.get("top_itemsets", []),
            pattern_insights=results.get("pattern_insights", {}),
            status="completed",
        )

        db_session.add(discovery_record)
        await db_session.commit()

        logger.info(f"Stored Apriori analysis metadata for run {discovery_run_id}")

    except Exception as e:
        logger.error(f"Error storing Apriori analysis metadata: {e}")
        await db_session.rollback()


async def _store_pattern_discovery_results(
    db_session: DBSession, discovery_run_id: str, results: dict[str, Any]
) -> None:
    """Store comprehensive pattern discovery results"""
    try:
        from ..database.models import AdvancedPatternResults

        # Extract metadata
        metadata = results.get("discovery_metadata", {})

        # Create results record
        results_record = AdvancedPatternResults(  # type: ignore[call-arg]
            discovery_run_id=discovery_run_id,
            algorithms_used=metadata.get("algorithms_used", []),
            discovery_modes=metadata.get("discovery_modes", []),
            parameter_patterns=results.get("traditional_patterns"),
            sequence_patterns=results.get("advanced_patterns", {}).get(
                "sequence_patterns"
            ),
            performance_patterns=results.get("advanced_patterns", {}).get(
                "performance_patterns"
            ),
            semantic_patterns=results.get("advanced_patterns", {}).get(
                "semantic_patterns"
            ),
            apriori_patterns=results.get("apriori_patterns"),
            cross_validation=results.get("cross_validation"),
            ensemble_analysis=results.get("advanced_patterns", {}).get(
                "ensemble_analysis"
            ),
            unified_recommendations=results.get("unified_recommendations", []),
            business_insights=results.get("business_insights", {}),
            execution_time_seconds=metadata.get("execution_time_seconds", 0.0),
            total_patterns_discovered=metadata.get("total_patterns_discovered", 0),
            discovery_quality_score=metadata.get("discovery_quality_score"),
            algorithms_count=metadata.get("algorithms_count", 1),
        )

        db_session.add(results_record)
        await db_session.commit()

        logger.info(f"Stored pattern discovery results for run {discovery_run_id}")

    except Exception as e:
        logger.error(f"Error storing pattern discovery results: {e}")
        await db_session.rollback()
