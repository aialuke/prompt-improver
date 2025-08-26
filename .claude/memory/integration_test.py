#!/usr/bin/env python3
"""Integration test for the complete agent memory system workflow."""

import sys

from memory_hooks import AgentMemoryHooks


def test_complete_workflow():
    """Test the complete memory workflow."""
    print("🧪 Testing Complete Agent Memory Workflow\n")

    # Initialize hooks
    hooks = AgentMemoryHooks()

    # Test Case 1: Database Query Optimization
    print("=" * 60)
    print("TEST CASE 1: Database Query Optimization")
    print("=" * 60)

    task_prompt = "Optimize the slow PostgreSQL query in the analytics dashboard that takes 5 seconds"

    # Pre-task: Load memory
    print("📥 PRE-TASK: Loading agent memory...")
    context = hooks.pre_task_memory_load(task_prompt)

    if context["memory_loaded"]:
        agent_name = context["agent_name"]
        task_id = context["task_id"]
        print(f"✅ Agent detected: {agent_name}")
        print(f"✅ Task ID: {task_id}")
        print(f"✅ Recent tasks: {len(context['context']['recent_tasks'])}")
        print(f"✅ Optimization insights: {len(context['context']['optimization_insights'])}")
        print(f"✅ Unread messages: {len(context['context']['unread_messages'])}")

        # Simulate task execution and completion
        print("\n⚙️  TASK EXECUTION: [Simulated] Query analysis and optimization...")

        # Post-task: Update memory
        print("\n📤 POST-TASK: Updating agent memory...")
        success = hooks.post_task_memory_update(
            task_outcome="success",
            insights=[
                "Added EXPLAIN analysis to identify bottleneck",
                "Created covering index on (user_id, created_at, status)",
                "Query response time reduced from 5.2s to 0.4s (92% improvement)"
            ],
            delegations=[
                {"to_agent": "performance-engineer", "reason": "system-wide performance validation", "outcome": "success"}
            ],
            optimization_area="query_optimization",
            optimization_insight="Covering index eliminated table scan for analytics query filtering",
            optimization_impact="high",
            optimization_confidence=0.98
        )
        print(f"✅ Memory update successful: {success}")

        # Send insight to performance engineer
        message_id = hooks.send_cross_agent_insight(
            message_type="insight",
            content="Database query optimization achieved 92% performance improvement, affecting dashboard load times",
            target_agents=["performance-engineer"],
            priority="high",
            metadata={"performance_gain": "92%", "affected_component": "analytics_dashboard"}
        )
        print(f"✅ Insight sent to performance-engineer: {message_id[:8] if message_id else 'Failed'}...")

    else:
        print("❌ Failed to load agent memory")
        return False

    # Test Case 2: ML Model Training
    print("\n" + "=" * 60)
    print("TEST CASE 2: ML Model Training")
    print("=" * 60)

    ml_prompt = "Optimize machine learning model training for better accuracy in the prompt improvement system"

    # Pre-task: Load ML orchestrator memory
    print("📥 PRE-TASK: Loading ML orchestrator memory...")
    ml_context = hooks.pre_task_memory_load(ml_prompt)

    if ml_context["memory_loaded"]:
        ml_agent = ml_context["agent_name"]
        print(f"✅ ML Agent detected: {ml_agent}")

        # Simulate ML task completion
        print("\n⚙️  TASK EXECUTION: [Simulated] Hyperparameter tuning and model optimization...")

        ml_success = hooks.post_task_memory_update(
            task_outcome="success",
            insights=[
                "Hyperparameter tuning improved model accuracy from 84% to 91%",
                "Feature engineering with context-aware learning patterns",
                "Model training time reduced by 35% with optimized batch processing"
            ],
            delegations=[
                {"to_agent": "data-pipeline-specialist", "reason": "feature engineering optimization", "outcome": "success"},
                {"to_agent": "infrastructure-specialist", "reason": "model deployment", "outcome": "success"}
            ],
            optimization_area="training_optimization",
            optimization_insight="Context-aware learning with domain-specific feature extraction",
            optimization_impact="high",
            optimization_confidence=0.94
        )
        print(f"✅ ML Memory update successful: {ml_success}")

    # Test Case 3: Cross-Agent Collaboration Test
    print("\n" + "=" * 60)
    print("TEST CASE 3: Cross-Agent Collaboration Pattern Analysis")
    print("=" * 60)

    # Get collaboration recommendations for database-specialist
    recommendations = hooks.get_collaboration_recommendations("database-specialist")
    print("🤝 Collaboration Recommendations for database-specialist:")

    for collab in recommendations.get("most_effective_collaborators", []):
        print(f"  • {collab['agent']}: {int(collab['success_rate'] * 100)}% success ({collab['frequency']} times)")
        if collab.get("common_tasks"):
            print(f"    Best for: {', '.join(collab['common_tasks'])}")

    for suggestion in recommendations.get("delegation_suggestions", []):
        print(f"  → Recommend delegating to {suggestion['delegate_to']} (Success: {int(suggestion['success_rate'] * 100)}%)")

    # Test memory cleanup
    print("\n" + "=" * 60)
    print("TEST CASE 4: Memory Cleanup")
    print("=" * 60)

    print("🧹 Testing memory cleanup (keeping last 30 days)...")
    cleaned_count = hooks.manager.cleanup_expired_data(days_to_keep=30)
    print(f"✅ Cleaned up {cleaned_count} expired records")

    return True


def test_agent_detection_accuracy():
    """Test agent detection accuracy across various prompts."""
    print("\n🎯 Testing Agent Detection Accuracy")
    print("=" * 50)

    hooks = AgentMemoryHooks()

    test_cases = [
        ("Optimize slow PostgreSQL query in analytics dashboard", "database-specialist"),
        ("Train machine learning model for better accuracy", "ml-orchestrator"),
        ("API endpoint responding slowly, need performance analysis", "performance-engineer"),
        ("Review authentication security for JWT token handling", "security-architect"),
        ("Set up Docker containers for testing environment", "infrastructure-specialist"),
        ("Design FastAPI endpoints for user management", "api-design-specialist"),
        ("Configure OpenTelemetry distributed tracing", "monitoring-observability-specialist"),
        ("Implement real behavior testing with testcontainers", "testing-strategy-specialist"),
        ("Set up environment configuration management", "configuration-management-specialist"),
        ("Create API documentation and architecture decisions", "documentation-specialist"),
        ("Optimize data pipeline ETL performance", "data-pipeline-specialist")
    ]

    correct = 0
    for prompt, expected_agent in test_cases:
        detected_agent = hooks.detect_agent_from_context(prompt)
        status = "✅" if detected_agent == expected_agent else "❌"
        print(f"{status} '{prompt[:40]}...' → {detected_agent} (expected: {expected_agent})")
        if detected_agent == expected_agent:
            correct += 1

    accuracy = (correct / len(test_cases)) * 100
    print(f"\n📊 Detection Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")

    return accuracy >= 80  # 80% accuracy threshold


if __name__ == "__main__":
    print("🚀 Starting Agent Memory System Integration Tests\n")

    # Test agent detection
    detection_success = test_agent_detection_accuracy()

    # Test complete workflow
    workflow_success = test_complete_workflow()

    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Agent Detection: {'✅ PASSED' if detection_success else '❌ FAILED'}")
    print(f"Complete Workflow: {'✅ PASSED' if workflow_success else '❌ FAILED'}")

    if detection_success and workflow_success:
        print("\n🎉 ALL TESTS PASSED - Agent Memory System Ready!")
        print("\n📋 Next Steps:")
        print("  1. Integrate with Claude Code's Task tool")
        print("  2. Set up automatic hooks in agent delegation workflow")
        print("  3. Configure memory cleanup schedules")
        print("  4. Monitor agent memory effectiveness")
    else:
        print("\n⚠️  Some tests failed - review implementation")

    sys.exit(0 if detection_success and workflow_success else 1)
