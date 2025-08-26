#!/usr/bin/env python3
"""Advanced Shared Context System.

Provides sophisticated cross-agent communication, project-wide intelligence sharing,
dynamic context evolution, and collaborative decision-making capabilities.
"""

import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from memory_manager import AgentMemoryManager


class SharedContextSystem:
    """Advanced shared context management for cross-agent collaboration."""

    def __init__(self, memory_dir: str = ".claude/memory") -> None:
        """Initialize shared context system."""
        self.manager = AgentMemoryManager(memory_dir)
        self.memory_dir = Path(memory_dir)

        # Agent categories for intelligent routing
        self.agent_categories = {
            "infrastructure": ["database-specialist", "infrastructure-specialist", "performance-engineer"],
            "intelligence": ["ml-orchestrator", "data-pipeline-specialist", "monitoring-observability-specialist"],
            "application": ["api-design-specialist", "testing-strategy-specialist", "configuration-management-specialist"],
            "governance": ["security-architect", "documentation-specialist"]
        }

        # Context evolution tracking
        self.context_evolution_patterns = {
            "performance_trends": [],
            "architectural_decisions": [],
            "collaboration_networks": {},
            "knowledge_convergence": {}
        }

    def analyze_project_context(self) -> dict[str, Any]:
        """Analyze current project context across all agents."""
        context = self.manager.load_shared_context()

        analysis = {
            "project_health": self._assess_project_health(context),
            "agent_ecosystem_status": self._analyze_agent_ecosystem(),
            "knowledge_synthesis": self._synthesize_cross_agent_knowledge(),
            "collaboration_network": self._analyze_collaboration_network(),
            "emerging_patterns": self._identify_emerging_patterns(),
            "recommended_actions": []
        }

        # Generate recommendations based on analysis
        analysis["recommended_actions"] = self._generate_context_recommendations(analysis)

        return analysis

    def create_global_insight(self, category: str, insight: str, contributing_agents: list[str],
                            confidence: float = 0.85, impact_level: str = "medium",
                            evidence: list[str] | None = None) -> str:
        """Create a new global insight accessible to all agents."""
        context = self.manager.load_shared_context()

        insight_id = str(uuid.uuid4())

        global_insight = {
            "insight_id": insight_id,
            "category": category,
            "insight": insight,
            "contributing_agents": contributing_agents,
            "confidence": confidence,
            "impact_assessment": impact_level,
            "created_at": datetime.now(UTC).isoformat() + "Z",
            "last_validated": datetime.now(UTC).isoformat() + "Z",
            "evidence": evidence or [],
            "validation_count": 0,
            "challenges": [],
            "evolution_history": []
        }

        context["global_insights"].append(global_insight)

        # Keep only last 20 global insights
        context["global_insights"] = context["global_insights"][-20:]

        self.manager.save_shared_context(context)

        # Broadcast insight to relevant agents
        self._broadcast_global_insight(insight_id, global_insight)

        return insight_id

    def create_architectural_decision(self, title: str, decision: str, rationale: str,
                                    alternatives: list[str], impact_areas: list[str],
                                    decided_by: list[str]) -> str:
        """Create a new architectural decision record."""
        context = self.manager.load_shared_context()

        decision_id = str(uuid.uuid4())

        architectural_decision = {
            "decision_id": decision_id,
            "title": title,
            "decision": decision,
            "rationale": rationale,
            "alternatives_considered": alternatives,
            "impact_areas": impact_areas,
            "decided_by": decided_by,
            "decision_date": datetime.now(UTC).isoformat() + "Z",
            "review_date": (datetime.now(UTC) + timedelta(days=90)).isoformat() + "Z",
            "status": "active",
            "implementation_progress": {},
            "lessons_learned": []
        }

        context["architectural_decisions"].append(architectural_decision)

        # Keep only last 15 architectural decisions
        context["architectural_decisions"] = context["architectural_decisions"][-15:]

        self.manager.save_shared_context(context)

        # Notify relevant agents
        affected_agents = self._identify_affected_agents(impact_areas)
        self._notify_agents_of_decision(decision_id, architectural_decision, affected_agents)

        return decision_id

    def update_performance_baselines(self, performance_data: dict[str, Any],
                                   source_agent: str) -> bool:
        """Update system-wide performance baselines."""
        context = self.manager.load_shared_context()

        # Update performance baselines
        if "performance_baselines" not in context:
            context["performance_baselines"] = {}

        for category, metrics in performance_data.items():
            if category not in context["performance_baselines"]:
                context["performance_baselines"][category] = {}

            for metric, value in metrics.items():
                # Track trend
                old_value = context["performance_baselines"][category].get(metric)
                context["performance_baselines"][category][metric] = value

                if old_value is not None:
                    # Calculate improvement/degradation
                    change_percent = ((value - old_value) / old_value) * 100 if old_value != 0 else 0

                    if abs(change_percent) > 10:  # Significant change
                        self._record_performance_trend(category, metric, old_value, value,
                                                     change_percent, source_agent)

        context["performance_baselines"]["last_baseline_update"] = datetime.now(UTC).isoformat() + "Z"

        self.manager.save_shared_context(context)
        return True

    def create_collaboration_opportunity(self, primary_agent: str, target_agents: list[str],
                                       task_type: str, expected_outcome: str,
                                       priority: str = "medium") -> str:
        """Create a collaboration opportunity for agents."""
        opportunity_id = str(uuid.uuid4())

        message_content = f"""
🤝 **COLLABORATION OPPORTUNITY**

**Primary Agent:** {primary_agent}
**Target Agents:** {', '.join(target_agents)}
**Task Type:** {task_type}
**Expected Outcome:** {expected_outcome}
**Priority:** {priority.upper()}

**Recommended Approach:**
Based on historical collaboration patterns, this task would benefit from coordinated effort.
Consider establishing a collaboration workflow with clear responsibilities and success metrics.
        """.strip()

        message_id = self.manager.add_inter_agent_message(
            from_agent="shared-context-system",
            message_type="collaboration",
            content=message_content,
            target_agents=target_agents,
            metadata={
                "priority": priority,
                "collaboration_id": opportunity_id,
                "primary_agent": primary_agent,
                "task_type": task_type
            }
        )

        return opportunity_id

    def evolve_context_intelligence(self) -> dict[str, Any]:
        """Evolve shared context intelligence based on agent activities."""
        all_agents = [
            "database-specialist", "ml-orchestrator", "performance-engineer",
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist",
            "testing-strategy-specialist", "configuration-management-specialist",
            "documentation-specialist"
        ]

        evolution_results = {
            "knowledge_synthesis": [],
            "pattern_discoveries": [],
            "optimization_opportunities": [],
            "risk_assessments": [],
            "collaboration_improvements": []
        }

        # Analyze cross-agent patterns
        agent_insights = {}
        agent_collaborations = defaultdict(list)

        for agent in all_agents:
            try:
                memory = self.manager.load_agent_memory(agent)
                agent_insights[agent] = memory.get("optimization_insights", [])

                # Analyze collaboration patterns
                for task in memory.get("task_history", []):
                    for delegation in task.get("delegations", []):
                        if delegation.get("outcome") == "success":
                            agent_collaborations[agent].append(delegation["to_agent"])
            except:
                continue

        # Synthesize knowledge patterns
        insight_patterns = defaultdict(list)
        for agent, insights in agent_insights.items():
            for insight in insights:
                area = insight.get("area", "unknown")
                if insight.get("confidence", 0) > 0.8:
                    insight_patterns[area].append({
                        "agent": agent,
                        "insight": insight["insight"],
                        "confidence": insight["confidence"],
                        "impact": insight.get("impact", "medium")
                    })

        # Identify cross-agent knowledge synthesis opportunities
        for area, insights_list in insight_patterns.items():
            if len(insights_list) >= 2:  # Multiple agents have insights in this area
                agents = [i["agent"] for i in insights_list]
                avg_confidence = sum(i["confidence"] for i in insights_list) / len(insights_list)

                evolution_results["knowledge_synthesis"].append({
                    "area": area,
                    "contributing_agents": agents,
                    "confidence": avg_confidence,
                    "synthesis_opportunity": f"Multiple agents ({', '.join(agents)}) have high-confidence insights in {area.replace('_', ' ')}"
                })

        # Analyze collaboration networks
        collaboration_strength = {}
        for agent, collaborators in agent_collaborations.items():
            if collaborators:
                most_frequent = max(set(collaborators), key=collaborators.count)
                frequency = collaborators.count(most_frequent)

                if frequency >= 2:  # Strong collaboration pattern
                    evolution_results["collaboration_improvements"].append({
                        "primary_agent": agent,
                        "preferred_collaborator": most_frequent,
                        "success_frequency": frequency,
                        "recommendation": f"Consider formalizing {agent} → {most_frequent} collaboration pattern"
                    })

        # Identify optimization opportunities
        high_impact_areas = defaultdict(int)
        for agent, insights in agent_insights.items():
            for insight in insights:
                if insight.get("impact") == "high":
                    high_impact_areas[insight.get("area", "unknown")] += 1

        for area, count in high_impact_areas.items():
            if count >= 2:  # Multiple high-impact optimizations in same area
                evolution_results["optimization_opportunities"].append({
                    "area": area,
                    "optimization_count": count,
                    "recommendation": f"Focus optimization efforts on {area.replace('_', ' ')} - {count} high-impact improvements identified"
                })

        # Store evolution results in shared context
        context = self.manager.load_shared_context()
        context["context_evolution"] = {
            "last_evolution": datetime.now(UTC).isoformat() + "Z",
            "evolution_results": evolution_results
        }
        self.manager.save_shared_context(context)

        return evolution_results

    def get_agent_recommendations(self, agent_name: str) -> list[dict[str, Any]]:
        """Get personalized recommendations for a specific agent."""
        try:
            agent_memory = self.manager.load_agent_memory(agent_name)
            shared_context = self.manager.load_shared_context()

            # Collaboration recommendations
            collaborators = agent_memory.get("collaboration_patterns", {}).get("frequent_collaborators", [])
            recommendations = [{
                        "type": "collaboration",
                        "priority": "high",
                        "recommendation": f"Continue leveraging {collab['agent_name']} collaboration (proven {int(collab['success_rate'] * 100)}% success rate)",
                        "evidence": f"{collab['collaboration_frequency']} successful collaborations"
                    } for collab in collaborators if collab.get("success_rate", 0) > 0.85 and collab.get("collaboration_frequency", 0) >= 3]

            # Learning opportunities from global insights
            recommendations.extend({
                        "type": "learning",
                        "priority": "medium",
                        "recommendation": f"Explore {insight['category']} insights from {', '.join(insight['contributing_agents'])}",
                        "evidence": f"High-confidence ({int(insight['confidence'] * 100)}%) global insight available"
                    } for insight in shared_context.get("global_insights", []) if agent_name not in insight.get("contributing_agents", []) and
                    insight.get("confidence", 0) > 0.9 and
                    insight.get("impact_assessment") == "high")

            # Architectural alignment recommendations
            recommendations.extend({
                        "type": "architecture",
                        "priority": "high",
                        "recommendation": f"Align with architectural decision: {decision['title']}",
                        "evidence": f"Decision affects {', '.join(decision['impact_areas'])}"
                    } for decision in shared_context.get("architectural_decisions", []) if agent_name in self._identify_affected_agents(decision.get("impact_areas", [])))

            return recommendations[:5]  # Top 5 recommendations

        except Exception as e:
            return [{"type": "error", "recommendation": f"Could not generate recommendations: {e!s}"}]

    def _assess_project_health(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assess overall project health based on shared context."""
        health_score = 0
        factors = {}

        # Check system performance
        perf_baselines = context.get("performance_baselines", {})
        if perf_baselines:
            db_perf = perf_baselines.get("database_performance", {})
            if db_perf.get("cache_hit_rate", 0) > 0.9:
                health_score += 20
                factors["database_performance"] = "excellent"
            elif db_perf.get("cache_hit_rate", 0) > 0.7:
                health_score += 15
                factors["database_performance"] = "good"
            else:
                health_score += 5
                factors["database_performance"] = "needs_improvement"

        # Check agent collaboration activity
        messages = context.get("inter_agent_messages", [])
        recent_messages = [m for m in messages if self._is_recent(m.get("timestamp", ""))]
        if len(recent_messages) > 3:
            health_score += 25
            factors["agent_collaboration"] = "active"
        elif len(recent_messages) > 1:
            health_score += 15
            factors["agent_collaboration"] = "moderate"
        else:
            health_score += 5
            factors["agent_collaboration"] = "low"

        # Check knowledge evolution
        insights = context.get("global_insights", [])
        high_confidence_insights = [i for i in insights if i.get("confidence", 0) > 0.9]
        if len(high_confidence_insights) >= 3:
            health_score += 25
            factors["knowledge_quality"] = "high"
        elif len(high_confidence_insights) >= 1:
            health_score += 15
            factors["knowledge_quality"] = "moderate"
        else:
            health_score += 5
            factors["knowledge_quality"] = "developing"

        # Check architectural governance
        decisions = context.get("architectural_decisions", [])
        if len(decisions) >= 2:
            health_score += 20
            factors["architectural_governance"] = "active"
        elif len(decisions) >= 1:
            health_score += 10
            factors["architectural_governance"] = "emerging"
        else:
            health_score += 5
            factors["architectural_governance"] = "minimal"

        # Overall assessment
        if health_score >= 80:
            overall_status = "excellent"
        elif health_score >= 60:
            overall_status = "good"
        elif health_score >= 40:
            overall_status = "fair"
        else:
            overall_status = "needs_attention"

        return {
            "overall_status": overall_status,
            "health_score": health_score,
            "factors": factors,
            "recommendations": self._generate_health_recommendations(factors)
        }

    def _analyze_agent_ecosystem(self) -> dict[str, Any]:
        """Analyze the health and activity of the agent ecosystem."""
        all_agents = [
            "database-specialist", "ml-orchestrator", "performance-engineer",
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist",
            "testing-strategy-specialist", "configuration-management-specialist",
            "documentation-specialist"
        ]

        ecosystem_status = {
            "active_agents": 0,
            "total_agents": len(all_agents),
            "agent_activity_levels": {},
            "collaboration_density": 0,
            "knowledge_distribution": {}
        }

        total_collaborations = 0
        total_insights = 0

        for agent in all_agents:
            try:
                memory = self.manager.load_agent_memory(agent)

                # Count recent activity
                recent_tasks = [t for t in memory.get("task_history", [])
                              if self._is_recent(t.get("timestamp", ""))]

                if recent_tasks:
                    ecosystem_status["active_agents"] += 1
                    ecosystem_status["agent_activity_levels"][agent] = len(recent_tasks)

                # Count collaborations
                for task in memory.get("task_history", []):
                    total_collaborations += len(task.get("delegations", []))

                # Count insights
                insights = memory.get("optimization_insights", [])
                total_insights += len(insights)
                ecosystem_status["knowledge_distribution"][agent] = len(insights)

            except:
                ecosystem_status["agent_activity_levels"][agent] = 0

        if ecosystem_status["active_agents"] > 0:
            ecosystem_status["collaboration_density"] = total_collaborations / ecosystem_status["active_agents"]

        return ecosystem_status

    def _synthesize_cross_agent_knowledge(self) -> list[dict[str, Any]]:
        """Synthesize knowledge patterns across agents."""
        # This is already implemented in evolve_context_intelligence
        # Return recent synthesis results
        context = self.manager.load_shared_context()
        evolution = context.get("context_evolution", {})
        return evolution.get("evolution_results", {}).get("knowledge_synthesis", [])

    def _analyze_collaboration_network(self) -> dict[str, Any]:
        """Analyze the collaboration network between agents."""
        all_agents = [
            "database-specialist", "ml-orchestrator", "performance-engineer",
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist",
            "testing-strategy-specialist", "configuration-management-specialist",
            "documentation-specialist"
        ]

        collaboration_matrix = defaultdict(lambda: defaultdict(int))
        collaboration_success = defaultdict(lambda: defaultdict(list))

        for agent in all_agents:
            try:
                memory = self.manager.load_agent_memory(agent)
                collaborators = memory.get("collaboration_patterns", {}).get("frequent_collaborators", [])

                for collab in collaborators:
                    target = collab["agent_name"]
                    frequency = collab.get("collaboration_frequency", 0)
                    success_rate = collab.get("success_rate", 0)

                    collaboration_matrix[agent][target] = frequency
                    collaboration_success[agent][target].append(success_rate)
            except:
                continue

        # Find strongest collaboration pairs
        strong_pairs = []
        for agent, targets in collaboration_matrix.items():
            for target, frequency in targets.items():
                if frequency >= 3:  # Strong collaboration
                    avg_success = sum(collaboration_success[agent][target]) / len(collaboration_success[agent][target])
                    strong_pairs.append({
                        "primary": agent,
                        "secondary": target,
                        "frequency": frequency,
                        "success_rate": avg_success
                    })

        # Sort by collaboration strength
        strong_pairs.sort(key=lambda x: x["frequency"] * x["success_rate"], reverse=True)

        return {
            "collaboration_matrix": dict(collaboration_matrix),
            "strongest_partnerships": strong_pairs[:5],
            "network_density": len(strong_pairs) / (len(all_agents) * (len(all_agents) - 1)) if len(all_agents) > 1 else 0
        }

    def _identify_emerging_patterns(self) -> list[dict[str, Any]]:
        """Identify emerging patterns across the agent ecosystem."""
        patterns = []

        # Pattern 1: Convergent optimization areas
        all_agents = [
            "database-specialist", "ml-orchestrator", "performance-engineer",
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist",
            "testing-strategy-specialist", "configuration-management-specialist",
            "documentation-specialist"
        ]

        optimization_areas = defaultdict(list)

        for agent in all_agents:
            try:
                memory = self.manager.load_agent_memory(agent)
                for insight in memory.get("optimization_insights", []):
                    area = insight.get("area", "unknown")
                    optimization_areas[area].append(agent)
            except:
                continue

        # Find areas with multiple agent involvement
        for area, agents in optimization_areas.items():
            if len(agents) >= 2:
                patterns.append({
                    "type": "convergent_optimization",
                    "pattern": f"Multiple agents optimizing {area.replace('_', ' ')}",
                    "agents_involved": agents,
                    "strength": len(agents)
                })

        return patterns

    def _generate_context_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations based on context analysis."""
        recommendations = []

        project_health = analysis.get("project_health", {})
        if project_health.get("health_score", 0) < 60:
            recommendations.append("Focus on improving project health metrics - consider increasing agent collaboration")

        ecosystem = analysis.get("agent_ecosystem_status", {})
        if ecosystem.get("active_agents", 0) < ecosystem.get("total_agents", 0) * 0.5:
            recommendations.append("Low agent ecosystem activity - consider activating more specialized agents")

        collaboration = analysis.get("collaboration_network", {})
        if collaboration.get("network_density", 0) < 0.1:
            recommendations.append("Sparse collaboration network - create more cross-agent collaboration opportunities")

        patterns = analysis.get("emerging_patterns", [])
        if len(patterns) >= 2:
            recommendations.append("Strong emerging patterns detected - consider formalizing successful collaboration workflows")

        return recommendations

    def _broadcast_global_insight(self, insight_id: str, insight: dict[str, Any]) -> None:
        """Broadcast global insight to all relevant agents."""
        content = f"Global Insight ({insight['category']}): {insight['insight']}"

        self.manager.add_inter_agent_message(
            from_agent="shared-context-system",
            message_type="insight",
            content=content,
            target_agents=[],  # Broadcast to all
            metadata={
                "insight_id": insight_id,
                "confidence": insight["confidence"],
                "impact_level": insight["impact_assessment"],
                "priority": "high" if insight["impact_assessment"] == "high" else "medium"
            }
        )

    def _identify_affected_agents(self, impact_areas: list[str]) -> list[str]:
        """Identify agents affected by impact areas."""
        affected = []

        area_mapping = {
            "database": ["database-specialist"],
            "performance": ["performance-engineer", "database-specialist"],
            "security": ["security-architect"],
            "infrastructure": ["infrastructure-specialist"],
            "ml": ["ml-orchestrator"],
            "api": ["api-design-specialist"],
            "testing": ["testing-strategy-specialist"],
            "configuration": ["configuration-management-specialist"],
            "documentation": ["documentation-specialist"],
            "monitoring": ["monitoring-observability-specialist"],
            "data": ["data-pipeline-specialist"]
        }

        for area in impact_areas:
            area_lower = area.lower()
            for key, agents in area_mapping.items():
                if key in area_lower:
                    affected.extend(agents)

        return list(set(affected))  # Remove duplicates

    def _notify_agents_of_decision(self, decision_id: str, decision: dict[str, Any],
                                 affected_agents: list[str]) -> None:
        """Notify agents of architectural decisions."""
        content = f"Architectural Decision: {decision['title']} - {decision['decision'][:100]}..."

        self.manager.add_inter_agent_message(
            from_agent="shared-context-system",
            message_type="context",
            content=content,
            target_agents=affected_agents,
            metadata={
                "decision_id": decision_id,
                "impact_areas": decision["impact_areas"],
                "priority": "high"
            }
        )

    def _record_performance_trend(self, category: str, metric: str, old_value: float,
                                new_value: float, change_percent: float, source_agent: str) -> None:
        """Record significant performance trends."""
        trend_type = "improvement" if change_percent > 0 else "degradation"

        content = f"Performance {trend_type}: {category}.{metric} changed {change_percent:+.1f}% ({old_value} → {new_value})"

        self.manager.add_inter_agent_message(
            from_agent=source_agent,
            message_type="insight",
            content=content,
            target_agents=[],  # Broadcast
            metadata={
                "trend_type": trend_type,
                "change_percent": change_percent,
                "category": category,
                "metric": metric,
                "priority": "high" if abs(change_percent) > 25 else "medium"
            }
        )

    def _is_recent(self, timestamp: str, hours: int = 24) -> bool:
        """Check if timestamp is within recent hours."""
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return (datetime.now(UTC).replace(tzinfo=ts.tzinfo) - ts).total_seconds() < hours * 3600
        except:
            return False

    def _generate_health_recommendations(self, factors: dict[str, str]) -> list[str]:
        """Generate health improvement recommendations."""
        recommendations = []

        if factors.get("database_performance") == "needs_improvement":
            recommendations.append("Optimize database performance - focus on query optimization and caching")

        if factors.get("agent_collaboration") == "low":
            recommendations.append("Increase agent collaboration - create more cross-agent workflows")

        if factors.get("knowledge_quality") == "developing":
            recommendations.append("Enhance knowledge quality - validate and synthesize insights across agents")

        if factors.get("architectural_governance") == "minimal":
            recommendations.append("Establish architectural governance - document key decisions and patterns")

        return recommendations


# Global shared context system instance
shared_context_system = SharedContextSystem()


def analyze_project_intelligence() -> dict[str, Any]:
    """Convenience function to analyze project intelligence."""
    return shared_context_system.analyze_project_context()


def create_global_project_insight(category: str, insight: str, contributing_agents: list[str],
                                confidence: float = 0.85, impact_level: str = "medium") -> str:
    """Convenience function to create global insights."""
    return shared_context_system.create_global_insight(
        category, insight, contributing_agents, confidence, impact_level
    )


def evolve_shared_intelligence() -> dict[str, Any]:
    """Convenience function to evolve shared context intelligence."""
    return shared_context_system.evolve_context_intelligence()


def get_personalized_agent_recommendations(agent_name: str) -> list[dict[str, Any]]:
    """Convenience function to get agent recommendations."""
    return shared_context_system.get_agent_recommendations(agent_name)


if __name__ == "__main__":
    # Test shared context system
    print("🌐 Testing Advanced Shared Context System")
    print("=" * 55)

    system = SharedContextSystem()

    # Test 1: Project intelligence analysis
    print("📊 Analyzing Project Intelligence...")
    intelligence = system.analyze_project_context()

    print(f"Project Health: {intelligence['project_health']['overall_status']} (Score: {intelligence['project_health']['health_score']})")
    print(f"Active Agents: {intelligence['agent_ecosystem_status']['active_agents']}/{intelligence['agent_ecosystem_status']['total_agents']}")
    print(f"Collaboration Network Density: {intelligence['collaboration_network']['network_density']:.3f}")

    # Test 2: Context evolution
    print("\n🧠 Evolving Context Intelligence...")
    evolution = system.evolve_context_intelligence()

    print(f"Knowledge Synthesis Opportunities: {len(evolution['knowledge_synthesis'])}")
    for synthesis in evolution['knowledge_synthesis'][:2]:
        print(f"  • {synthesis['area']}: {', '.join(synthesis['contributing_agents'])}")

    print(f"Optimization Opportunities: {len(evolution['optimization_opportunities'])}")
    for opt in evolution['optimization_opportunities'][:2]:
        print(f"  • {opt['area']}: {opt['optimization_count']} high-impact improvements")

    # Test 3: Create global insight
    print("\n💡 Creating Global Insight...")
    insight_id = system.create_global_insight(
        category="performance",
        insight="Agent memory system achieving 100% workflow success with 4.0 insights per workflow",
        contributing_agents=["database-specialist", "ml-orchestrator", "performance-engineer"],
        confidence=0.95,
        impact_level="high"
    )
    print(f"Global insight created: {insight_id[:8]}...")

    # Test 4: Agent recommendations
    print("\n🎯 Getting Agent Recommendations...")
    for agent in ["database-specialist", "ml-orchestrator"]:
        recommendations = system.get_agent_recommendations(agent)
        print(f"\n{agent} recommendations:")
        for rec in recommendations[:2]:
            print(f"  • {rec.get('type', 'general').title()}: {rec.get('recommendation', 'No recommendation')[:60]}...")

    print("\n✅ Shared context system test completed successfully!")
