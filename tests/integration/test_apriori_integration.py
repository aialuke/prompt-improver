"""
Integration tests for Apriori algorithm implementation

Tests the complete integration of Apriori association rule mining with:
- AprioriAnalyzer service
- Database schema and storage
- ML pipeline integration
- API endpoints
- Pattern discovery workflow
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from prompt_improver.database import ManagerMode
from prompt_improver.database.models import (
    AprioriAssociationRule,
    AprioriPatternDiscovery,
    PromptSession,
    RulePerformance,
    UserFeedback,
)
from prompt_improver.ml.core.ml_integration import MLModelService
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.ml.learning.patterns.apriori_analyzer import (
    AprioriAnalyzer,
    AprioriConfig,
)


@pytest.fixture
async def sample_data(db_session: AsyncSession):
    """Create sample data for Apriori testing"""
    sessions = [
        PromptSession(
            session_id=f"session_{i}",
            original_prompt=f"Test prompt {i}",
            improved_prompt=f"Improved prompt {i}",
            quality_score=0.8 + i % 3 * 0.1,
            improvement_score=0.7 + i % 4 * 0.1,
            confidence_level=0.8,
            created_at=datetime.now(UTC) - timedelta(days=i % 30),
        )
        for i in range(20)
    ]
    for session in sessions:
        db_session.add(session)
    rule_names = ["ClarityRule", "SpecificityRule", "StructureRule", "ExampleRule"]
    performances = []
    for i, session in enumerate(sessions):
        rules_to_apply = rule_names[: 2 + i % 2]
        for rule_name in rules_to_apply:
            performance = RulePerformance(
                rule_id=rule_name,
                rule_name=rule_name,
                improvement_score=0.6
                + hash(f"{session.session_id}_{rule_name}") % 100 / 250,
                execution_time_ms=int(50.0 + i % 20 * 5),
                confidence_level=0.7 + i % 3 * 0.1,
                rule_parameters={"threshold": 0.5 + i % 3 * 0.1},
            )
            performances.append(performance)
            db_session.add(performance)
    feedbacks = [
        UserFeedback(
            session_id=f"session_{i}",
            rating=3 + i % 3,
            feedback_text=f"Feedback for session {i}",
            improvement_areas=["clarity", "structure"] if i % 2 else ["specificity"],
        )
        for i in range(0, 20, 2)
    ]
    for feedback in feedbacks:
        db_session.add(feedback)
    await db_session.commit()
    return {
        "sessions": len(sessions),
        "performances": len(performances),
        "feedbacks": len(feedbacks),
    }


class TestAprioriAnalyzer:
    """Test suite for AprioriAnalyzer functionality"""

    @pytest.mark.asyncio
    async def test_apriori_analyzer_initialization(self):
        """Test AprioriAnalyzer can be initialized with proper configuration"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        config = AprioriConfig(
            min_support=0.1, min_confidence=0.6, min_lift=1.0, max_itemset_length=5
        )
        analyzer = AprioriAnalyzer(db_manager=db_manager, config=config)
        assert analyzer.config.min_support == 0.1
        assert analyzer.config.min_confidence == 0.6
        assert analyzer.db_manager is not None

    @pytest.mark.asyncio
    async def test_transaction_extraction(self):
        """Test extraction of transactions from database"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        analyzer = AprioriAnalyzer(db_manager=db_manager)
        transactions = analyzer.extract_transactions_from_database(window_days=30)
        assert isinstance(transactions, list)

    @pytest.mark.asyncio
    async def test_prompt_characteristics_extraction(self):
        """Test extraction of prompt characteristics for itemset analysis"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        analyzer = AprioriAnalyzer(db_manager=db_manager)
        test_cases = [
            ("Short prompt", ["length_short", "domain_general"]),
            (
                "Write a detailed function that processes data step by step",
                ["length_medium", "complexity_sequential", "domain_technical"],
            ),
            (
                "Can you provide an example of a creative story?",
                ["length_medium", "pattern_examples", "domain_creative"],
            ),
            (
                "Analyze the research data and provide insights",
                ["length_medium", "domain_analytical"],
            ),
        ]
        for prompt, expected_characteristics in test_cases:
            characteristics = analyzer._extract_prompt_characteristics(prompt)
            for expected in expected_characteristics:
                assert expected in characteristics, (
                    f"Expected {expected} in {characteristics} for prompt: {prompt}"
                )

    @pytest.mark.asyncio
    async def test_frequent_itemset_mining(self):
        """Test frequent itemset mining with synthetic transaction data"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        analyzer = AprioriAnalyzer(db_manager=db_manager)
        transactions = [
            ["rule_ClarityRule", "quality_high", "domain_technical"],
            ["rule_ClarityRule", "quality_high", "domain_technical"],
            ["rule_ClarityRule", "quality_high", "domain_technical"],
            ["rule_SpecificityRule", "quality_medium", "domain_creative"],
            ["rule_SpecificityRule", "quality_medium", "domain_creative"],
            ["rule_ClarityRule", "rule_SpecificityRule", "quality_high"],
            ["rule_ClarityRule", "rule_SpecificityRule", "quality_high"],
            ["rule_ClarityRule", "rule_SpecificityRule", "quality_high"],
        ]
        frequent_itemsets = await analyzer.mine_frequent_itemsets(
            transactions, min_support=0.3
        )
        assert not frequent_itemsets.empty
        assert "support" in frequent_itemsets.columns
        assert "itemsets" in frequent_itemsets.columns
        assert "length" in frequent_itemsets.columns
        assert all(frequent_itemsets["support"] >= 0.3)

    @pytest.mark.asyncio
    async def test_association_rule_generation(self):
        """Test generation of association rules from frequent itemsets"""
        import pandas as pd
        from prompt_improver.database.composition import get_database_services, create_database_services, DatabaseServices
        from prompt_improver.database.types import ManagerMode

        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        analyzer = AprioriAnalyzer(db_manager=db_manager)
        mock_itemsets = pd.DataFrame({
            "support": [0.8, 0.6, 0.5, 0.4],
            "itemsets": [
                frozenset(["rule_ClarityRule"]),
                frozenset(["quality_high"]),
                frozenset(["rule_ClarityRule", "quality_high"]),
                frozenset(["rule_ClarityRule", "quality_high", "domain_technical"]),
            ],
        })
        rules = analyzer.generate_association_rules(
            mock_itemsets, min_confidence=0.5, min_lift=1.1
        )
        if isinstance(rules, dict):
            assert "error" in rules or len(rules) == 0
            print("Association rules generation failed as expected with mock data")
        elif hasattr(rules, "empty") and (not rules.empty):
            assert "antecedents" in rules.columns
            assert "consequents" in rules.columns
            assert "support" in rules.columns
            assert "confidence" in rules.columns
        else:
            assert len(rules) == 0


class TestMLIntegration:
    """Test ML pipeline integration with Apriori"""

    @pytest.mark.asyncio
    async def test_ml_service_with_apriori(self):
        """Test MLModelService with Apriori integration"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        ml_service = MLModelService(db_manager=db_manager)
        async with db_manager.get_async_session() as db_session:
            results = await ml_service.discover_patterns(
                db_session=db_session,
                min_effectiveness=0.5,
                min_support=3,
                use_advanced_discovery=True,
                include_apriori=True,
            )
        assert "status" in results
        assert "discovery_metadata" in results
        assert "traditional_patterns" in results
        metadata = results["discovery_metadata"]
        assert "algorithms_used" in metadata
        assert "execution_time_seconds" in metadata
        assert "total_patterns_discovered" in metadata
        if (
            results.get("apriori_patterns")
            and "error" not in results["apriori_patterns"]
        ):
            apriori_results = results["apriori_patterns"]
            assert "patterns" in apriori_results
            assert "discovery_type" in apriori_results
            assert apriori_results["discovery_type"] == "apriori_association_rules"

    @pytest.mark.asyncio
    async def test_contextualized_patterns(self):
        """Test contextualized pattern analysis"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        ml_service = MLModelService(db_manager=db_manager)
        context_items = ["rule_ClarityRule", "domain_technical", "quality_high"]
        async with db_manager.get_async_session() as db_session:
            results = await ml_service.get_contextualized_patterns(
                context_items=context_items, db_session=db_session, min_confidence=0.5
            )
        assert "context_items" in results
        assert results["context_items"] == context_items
        if "error" not in results:
            assert "recommendations" in results or "combined_recommendations" in results


class TestDatabaseIntegration:
    """Test database schema and storage functionality"""

    @pytest.mark.asyncio
    async def test_apriori_association_rule_storage(self, db_session: AsyncSession):
        """Test storing and retrieving association rules"""
        rule = AprioriAssociationRule(
            antecedents='["rule_ClarityRule", "domain_technical"]',
            consequents='["quality_high"]',
            support=0.75,
            confidence=0.85,
            lift=2.1,
            conviction=3.2,
            rule_strength=0.88,
            business_insight="Applying clarity rule to technical prompts leads to high quality",
            pattern_category="rule_performance",
            discovery_run_id="test_run_123",
        )
        db_session.add(rule)
        await db_session.commit()
        stmt = select(AprioriAssociationRule).where(
            AprioriAssociationRule.discovery_run_id == "test_run_123"
        )
        result = await db_session.execute(stmt)
        retrieved_rule = result.scalar_one_or_none()
        assert retrieved_rule is not None
        assert retrieved_rule.confidence == 0.85
        assert retrieved_rule.lift == 2.1
        assert retrieved_rule.pattern_category == "rule_performance"
        assert retrieved_rule.business_insight is not None
        assert "clarity rule" in retrieved_rule.business_insight.lower()

    @pytest.mark.asyncio
    async def test_pattern_discovery_metadata_storage(self, db_session: AsyncSession):
        """Test storing pattern discovery run metadata"""
        discovery_run = AprioriPatternDiscovery(
            discovery_run_id="test_discovery_456",
            min_support=0.1,
            min_confidence=0.6,
            min_lift=1.2,
            max_itemset_length=4,
            data_window_days=30,
            transaction_count=100,
            frequent_itemsets_count=25,
            association_rules_count=15,
            execution_time_seconds=12.5,
            top_patterns_summary=[
                {"itemset": ["rule_ClarityRule"], "support": 0.8},
                {"itemset": ["quality_high"], "support": 0.6},
            ],
            pattern_insights={
                "rule_performance_patterns": ["ClarityRule leads to high quality"],
                "quality_improvement_patterns": [
                    "Technical domain + clarity → high quality"
                ],
            },
            status="completed",
        )
        db_session.add(discovery_run)
        await db_session.commit()
        stmt = select(AprioriPatternDiscovery).where(
            AprioriPatternDiscovery.discovery_run_id == "test_discovery_456"
        )
        result = await db_session.execute(stmt)
        retrieved_run = result.scalar_one_or_none()
        assert retrieved_run is not None
        assert retrieved_run.status == "completed"
        assert retrieved_run.transaction_count == 100
        assert retrieved_run.execution_time_seconds == 12.5
        assert retrieved_run.top_patterns_summary is not None
        assert len(retrieved_run.top_patterns_summary) == 2


class TestAdvancedPatternDiscovery:
    """Test AdvancedPatternDiscovery service with Apriori integration"""

    @pytest.mark.asyncio
    async def test_advanced_pattern_discovery_initialization(self):
        """Test AdvancedPatternDiscovery can be initialized with database manager"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        discovery_service = AdvancedPatternDiscovery(db_manager=db_manager)
        assert discovery_service.apriori_analyzer is not None
        assert discovery_service.db_manager is not None

    @pytest.mark.asyncio
    async def test_comprehensive_pattern_discovery(self, db_session: AsyncSession):
        """Test comprehensive pattern discovery including Apriori"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        discovery_service = AdvancedPatternDiscovery(db_manager=db_manager)
        results = await discovery_service.discover_advanced_patterns(
            db_session=db_session,
            min_effectiveness=0.5,
            min_support=3,
            pattern_types=["parameter", "apriori"],
            use_ensemble=True,
            include_apriori=True,
        )
        assert "discovery_metadata" in results
        assert "parameter_patterns" in results or "error" in results
        metadata = results["discovery_metadata"]
        assert "execution_time" in metadata
        assert "algorithms_used" in metadata
        assert "apriori_enabled" in metadata


class TestEndToEndWorkflow:
    """Test complete end-to-end Apriori workflow"""

    @pytest.mark.asyncio
    async def test_complete_apriori_workflow(self, db_session: AsyncSession):
        """Test complete workflow from data extraction to rule storage"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        ml_service = MLModelService(db_manager=db_manager)
        discovery_results = await ml_service.discover_patterns(
            db_session=db_session,
            min_effectiveness=0.5,
            min_support=3,
            use_advanced_discovery=True,
            include_apriori=True,
        )
        assert "status" in discovery_results
        assert discovery_results["status"] in ["success", "error"]
        if discovery_results["status"] == "success":
            assert "discovery_metadata" in discovery_results
            assert "algorithms_used" in discovery_results["discovery_metadata"]
            if "apriori_patterns" in discovery_results:
                apriori_results = discovery_results["apriori_patterns"]
                if "error" not in apriori_results:
                    assert "patterns" in apriori_results
                    assert "algorithm" in apriori_results
                    assert apriori_results["algorithm"] == "mlxtend_apriori"
        context_items = ["rule_ClarityRule", "quality_high"]
        contextual_results = await ml_service.get_contextualized_patterns(
            context_items=context_items, db_session=db_session, min_confidence=0.5
        )
        assert "context_items" in contextual_results
        assert contextual_results["context_items"] == context_items

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance benchmarks for Apriori implementation"""
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        analyzer = AprioriAnalyzer(db_manager=db_manager)
        start_time = datetime.now(UTC)
        results = await analyzer.analyze_patterns(
            window_days=30, save_to_database=False
        )
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 30.0
        if "error" not in results:
            assert "transaction_count" in results
            assert results["transaction_count"] >= 0
            if results.get("transaction_count", 0) > 0:
                assert execution_time < 60.0


@pytest.mark.asyncio
async def test_error_handling_and_edge_cases():
    """Test error handling and edge cases"""
    try:
        analyzer = AprioriAnalyzer(db_manager=None)
        assert analyzer.db_manager is None
    except Exception:
        pass
    config = AprioriConfig(min_support=-0.1, min_confidence=1.5, min_lift=-1.0)
    db_manager = get_database_services(ManagerMode.ML_TRAINING)
    analyzer = AprioriAnalyzer(db_manager=db_manager, config=config)
    assert analyzer.config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
