"""Test MCP-ML Architectural Separation.

Validates that the MCP-ML integration maintains proper architectural separation
without importing components that cause initialization issues.
"""

import pytest
import os
from pathlib import Path


class TestMCPMLArchitecturalSeparation:
    """Test architectural separation between MCP and ML systems."""

    def test_ml_engine_directory_removed(self):
        """Test that the incorrectly created ml_engine directory was removed."""
        project_root = Path(__file__).parent.parent
        ml_engine_path = project_root / "src" / "prompt_improver" / "ml_engine"

        assert not ml_engine_path.exists(), "ml_engine directory should be removed (architectural violation)"
        print("✅ ml_engine directory correctly removed - no architectural violation")

    def test_existing_ml_system_present(self):
        """Test that the existing ML system is present and should be used."""
        project_root = Path(__file__).parent.parent
        ml_path = project_root / "src" / "prompt_improver" / "ml"

        assert ml_path.exists(), "Existing ML system should be present"

        # Check for key existing ML components
        pattern_discovery = ml_path / "learning" / "patterns" / "advanced_pattern_discovery.py"
        rule_optimizer = ml_path / "optimization" / "algorithms" / "rule_optimizer.py"
        performance_calculator = ml_path / "analytics" / "performance_improvement_calculator.py"

        assert pattern_discovery.exists(), "AdvancedPatternDiscovery should exist"
        assert rule_optimizer.exists(), "RuleOptimizer should exist"
        assert performance_calculator.exists(), "PerformanceImprovementCalculator should exist"

        print("✅ Existing ML system components verified")

    def test_mcp_data_collector_file_exists(self):
        """Test that MCP data collector file exists."""
        project_root = Path(__file__).parent.parent
        data_collector_path = project_root / "src" / "prompt_improver" / "mcp_server" / "ml_data_collector.py"

        assert data_collector_path.exists(), "MCP ML data collector should exist"
        print("✅ MCP ML data collector file exists")

    def test_mcp_integration_file_exists(self):
        """Test that MCP integration file exists."""
        project_root = Path(__file__).parent.parent
        integration_path = project_root / "src" / "prompt_improver" / "mcp_server" / "ml_integration.py"

        assert integration_path.exists(), "MCP ML integration should exist"
        print("✅ MCP ML integration file exists")

    def test_mcp_data_collector_content_separation(self):
        """Test that MCP data collector only does data collection, not ML operations."""
        project_root = Path(__file__).parent.parent
        data_collector_path = project_root / "src" / "prompt_improver" / "mcp_server" / "ml_data_collector.py"

        with open(data_collector_path, 'r') as f:
            content = f.read()

        # Should contain data collection methods
        assert "collect_rule_application" in content, "Should have rule application collection"
        assert "collect_user_feedback" in content, "Should have user feedback collection"
        assert "prepare_ml_data_package" in content, "Should prepare data for ML system"

        # Should NOT contain ML operations
        ml_violations = [
            "def analyze_patterns",
            "def optimize_rules",
            "def generate_rules",
            "def calculate_performance",
            "class PatternRecognizer",
            "class RuleOptimizer",
            "class MLRuleGenerator"
        ]

        for violation in ml_violations:
            assert violation not in content, f"MCP should not contain ML operation: {violation}"

        print("✅ MCP data collector maintains architectural separation")

    def test_mcp_integration_content_separation(self):
        """Test that MCP integration only integrates with existing ML, doesn't do ML."""
        project_root = Path(__file__).parent.parent
        integration_path = project_root / "src" / "prompt_improver" / "mcp_server" / "ml_integration.py"

        with open(integration_path, 'r') as f:
            content = f.read()

        # Should import existing ML components
        assert "from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import AdvancedPatternDiscovery" in content
        assert "from prompt_improver.ml.optimization.algorithms.rule_optimizer import RuleOptimizer" in content
        assert "from prompt_improver.ml.analytics.performance_improvement_calculator import PerformanceImprovementCalculator" in content

        # Should have integration methods
        assert "get_ml_insights_for_prompt" in content, "Should request insights from ML system"
        assert "trigger_ml_analysis" in content, "Should trigger ML analysis"

        # Should NOT implement ML algorithms
        ml_violations = [
            "def _analyze_patterns",
            "def _optimize_rules_internal",
            "def _generate_rules_internal",
            "def _calculate_effectiveness_internal",
            "class InternalPatternRecognizer",
            "class InternalRuleOptimizer"
        ]

        for violation in ml_violations:
            assert violation not in content, f"MCP integration should not implement ML: {violation}"

        print("✅ MCP integration maintains architectural separation")

    def test_roadmap_reflects_correct_architecture(self):
        """Test that MCP_ROADMAP.md reflects the corrected architecture."""
        project_root = Path(__file__).parent.parent
        roadmap_path = project_root / "MCP_ROADMAP.md"

        with open(roadmap_path, 'r') as f:
            content = f.read()

        # Should mention architectural correction
        assert "Architecture Correction" in content or "Architectural Separation" in content
        assert "MCP = data collection only" in content or "Data collection and formatting only" in content

        # Should reference existing ML components
        assert "AdvancedPatternDiscovery" in content
        assert "rule_optimizer.py" in content
        assert "performance_improvement_calculator.py" in content

        # Should indicate Phase 2 completion with proper separation
        assert "Phase 2 Complete" in content or "PHASE 2 COMPLETE" in content

        print("✅ MCP roadmap reflects correct architectural separation")

    def test_multi_level_cache_integration(self):
        """Test that existing MultiLevelCache is referenced, not duplicated."""
        project_root = Path(__file__).parent.parent
        cache_path = project_root / "src" / "prompt_improver" / "utils" / "multi_level_cache.py"

        assert cache_path.exists(), "Existing MultiLevelCache should be present"

        # Check MCP integration references existing cache
        integration_path = project_root / "src" / "prompt_improver" / "mcp_server" / "ml_integration.py"
        with open(integration_path, 'r') as f:
            content = f.read()

        assert "from prompt_improver.utils.multi_level_cache import MultiLevelCache" in content

        print("✅ MCP integration uses existing MultiLevelCache")

    def test_postgresql_usage_consistency(self):
        """Test that PostgreSQL is used consistently, not SQLite."""
        project_root = Path(__file__).parent.parent

        # Check database configuration
        db_config_path = project_root / "src" / "prompt_improver" / "database" / "config.py"
        if db_config_path.exists():
            with open(db_config_path, 'r') as f:
                content = f.read()

            # Should reference PostgreSQL, not SQLite
            assert "postgresql" in content.lower() or "postgres" in content.lower()
            assert "sqlite" not in content.lower(), "Should not reference SQLite"

        print("✅ PostgreSQL usage verified")


if __name__ == "__main__":
    """Run architectural separation tests directly."""
    test_instance = TestMCPMLArchitecturalSeparation()

    print("🧪 Running MCP-ML Architectural Separation Tests...")

    try:
        test_instance.test_ml_engine_directory_removed()
        test_instance.test_existing_ml_system_present()
        test_instance.test_mcp_data_collector_file_exists()
        test_instance.test_mcp_integration_file_exists()
        test_instance.test_mcp_data_collector_content_separation()
        test_instance.test_mcp_integration_content_separation()
        test_instance.test_roadmap_reflects_correct_architecture()
        test_instance.test_multi_level_cache_integration()
        test_instance.test_postgresql_usage_consistency()

        print("🎉 All Architectural Separation Tests PASSED!")
        print("✅ MCP-ML architectural separation verified")
        print("✅ MCP role: Data collection and formatting only")
        print("✅ ML role: Pattern analysis, rule optimization, performance calculation")
        print("✅ Clean integration: MCP → existing ML system")
        print("✅ No duplication: Existing ML infrastructure reused")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
