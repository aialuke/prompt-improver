#!/usr/bin/env python3
"""Agent Metrics Tracker - Claude Code Agent Usage Monitor.

This script tracks and analyzes agent usage patterns by:
1. Monitoring which agents are invoked and for what tasks
2. Tracking agent effectiveness and user satisfaction patterns
3. Analyzing role boundary adherence and delegation patterns
4. Providing optimization recommendations for agent configurations
5. Detecting agent misuse or role boundary violations

Usage: Can be integrated as post-tool hook or run independently for analysis
"""

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class AgentUsageMetric:
    """Data class for agent usage metrics."""
    agent_name: str
    task_type: str
    timestamp: datetime
    duration_seconds: float | None = None
    success: bool = True
    user_prompt: str = ""
    delegation_pattern: list[str] = None
    tools_used: list[str] = None
    outcome_quality: str = "unknown"  # good, fair, poor, unknown
    boundary_violations: list[str] = None

    def __post_init__(self):
        if self.delegation_pattern is None:
            self.delegation_pattern = []
        if self.tools_used is None:
            self.tools_used = []
        if self.boundary_violations is None:
            self.boundary_violations = []


class AgentMetricsTracker:
    """Track and analyze agent usage metrics."""

    def __init__(self, metrics_file: str = ".claude/metrics/agent_usage.jsonl") -> None:
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        # Agent role definitions for boundary checking
        self.agent_roles = {
            "database-specialist": {
                "primary": ["query_optimization", "schema_design", "migrations", "indexing"],
                "delegates_to": ["performance-engineer"],
                "receives_from": ["performance-engineer", "infrastructure-specialist"]
            },
            "ml-orchestrator": {
                "primary": ["model_training", "ml_pipelines", "feature_engineering", "model_optimization"],
                "delegates_to": ["performance-engineer", "infrastructure-specialist"],
                "receives_from": ["data-pipeline-specialist"]
            },
            "performance-engineer": {
                "primary": ["performance_analysis", "bottleneck_identification", "monitoring_setup"],
                "delegates_to": ["database-specialist", "ml-orchestrator"],
                "receives_from": ["all"]
            },
            "security-architect": {
                "primary": ["security_design", "vulnerability_assessment", "auth_review"],
                "delegates_to": ["infrastructure-specialist"],
                "receives_from": []
            },
            "infrastructure-specialist": {
                "primary": ["containerization", "ci_cd", "environment_setup", "testcontainers"],
                "delegates_to": ["database-specialist"],
                "receives_from": ["security-architect", "performance-engineer"]
            }
        }

    def record_usage(self, metric: AgentUsageMetric) -> None:
        """Record agent usage metric to file."""
        metric_dict = asdict(metric)
        metric_dict['timestamp'] = metric.timestamp.isoformat()

        with open(self.metrics_file, 'a', encoding="utf-8") as f:
            f.write(json.dumps(metric_dict) + '\n')

    def load_metrics(self, days_back: int = 30) -> list[AgentUsageMetric]:
        """Load metrics from the last N days."""
        if not self.metrics_file.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=days_back)
        metrics = []

        with open(self.metrics_file, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])

                    if data['timestamp'] >= cutoff_date:
                        metric = AgentUsageMetric(**data)
                        metrics.append(metric)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Skipping invalid metric line: {e}", file=sys.stderr)
                    continue

        return metrics

    def analyze_usage_patterns(self, days_back: int = 7) -> dict[str, Any]:
        """Analyze agent usage patterns and effectiveness."""
        metrics = self.load_metrics(days_back)

        if not metrics:
            return {"error": "No metrics found"}

        analysis = {
            "total_invocations": len(metrics),
            "date_range": f"Last {days_back} days",
            "agent_usage": Counter([m.agent_name for m in metrics]),
            "task_types": Counter([m.task_type for m in metrics]),
            "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
            "average_duration": sum(m.duration_seconds or 0 for m in metrics) / len(metrics),
            "delegation_patterns": self._analyze_delegation_patterns(metrics),
            "boundary_violations": self._analyze_boundary_violations(metrics),
            "effectiveness_by_agent": self._analyze_effectiveness(metrics),
            "optimization_recommendations": []
        }

        # Generate optimization recommendations
        analysis["optimization_recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _analyze_delegation_patterns(self, metrics: list[AgentUsageMetric]) -> dict[str, Any]:
        """Analyze delegation patterns between agents."""
        delegations = defaultdict(list)

        for metric in metrics:
            if metric.delegation_pattern:
                for target in metric.delegation_pattern:
                    delegations[metric.agent_name].append(target)

        return {
            "delegation_frequency": {
                agent: Counter(targets)
                for agent, targets in delegations.items()
            },
            "most_common_delegations": [
                f"{agent} -> {target}"
                for agent, targets in delegations.items()
                for target, count in Counter(targets).most_common(1)
            ]
        }

    def _analyze_boundary_violations(self, metrics: list[AgentUsageMetric]) -> dict[str, Any]:
        """Analyze role boundary violations."""
        violations = defaultdict(list)

        for metric in metrics:
            if metric.boundary_violations:
                violations[metric.agent_name].extend(metric.boundary_violations)

        return {
            "violations_by_agent": dict(violations),
            "total_violations": sum(len(v) for v in violations.values()),
            "violation_rate": sum(len(v) for v in violations.values()) / len(metrics) if metrics else 0
        }

    def _analyze_effectiveness(self, metrics: list[AgentUsageMetric]) -> dict[str, dict[str, Any]]:
        """Analyze effectiveness by agent."""
        effectiveness = defaultdict(lambda: {
            "total_invocations": 0,
            "success_rate": 0,
            "avg_duration": 0,
            "quality_scores": Counter(),
            "common_tasks": Counter()
        })

        for metric in metrics:
            agent_stats = effectiveness[metric.agent_name]
            agent_stats["total_invocations"] += 1
            agent_stats["common_tasks"][metric.task_type] += 1
            agent_stats["quality_scores"][metric.outcome_quality] += 1

            if metric.duration_seconds:
                agent_stats["avg_duration"] += metric.duration_seconds

        # Calculate averages and rates
        for agent, stats in effectiveness.items():
            total = stats["total_invocations"]
            stats["success_rate"] = sum(1 for m in metrics if m.agent_name == agent and m.success) / total
            stats["avg_duration"] = stats["avg_duration"] / total if total > 0 else 0
            stats["most_common_task"] = stats["common_tasks"].most_common(1)[0] if stats["common_tasks"] else ("none", 0)

        return dict(effectiveness)

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Check success rates
        for agent, stats in analysis["effectiveness_by_agent"].items():
            if stats["success_rate"] < 0.8:
                recommendations.append(
                    f"🔍 {agent}: Low success rate ({stats['success_rate']:.1%}). "
                    f"Review role definition and provide additional guidance."
                )

        # Check delegation patterns
        delegation_freq = analysis["delegation_patterns"]["delegation_frequency"]
        for agent, delegations in delegation_freq.items():
            if len(delegations) > 3:
                recommendations.append(
                    f"⚡ {agent}: High delegation frequency. Consider expanding role scope "
                    f"or clarifying boundaries with {list(delegations.keys())[:2]}."
                )

        # Check boundary violations
        if analysis["boundary_violations"]["violation_rate"] > 0.1:
            recommendations.append(
                "🚨 High boundary violation rate. Review agent role definitions "
                "and improve delegation guidance."
            )

        # Check usage balance
        usage_counts = analysis["agent_usage"]
        max_usage = max(usage_counts.values()) if usage_counts else 0
        min_usage = min(usage_counts.values()) if usage_counts else 0

        if max_usage > min_usage * 3:  # Significant imbalance
            overused = max(usage_counts, key=usage_counts.get)
            underused = min(usage_counts, key=usage_counts.get)
            recommendations.append(
                f"⚖️ Usage imbalance: {overused} overused, {underused} underused. "
                f"Consider redistributing responsibilities or improving discoverability."
            )

        # Performance recommendations
        avg_duration = analysis["average_duration"]
        if avg_duration > 300:  # 5 minutes
            recommendations.append(
                f"⏱️ Average task duration is high ({avg_duration:.1f}s). "
                f"Consider optimizing agent configurations or task complexity."
            )

        return recommendations

    def generate_report(self, days_back: int = 7) -> str:
        """Generate a formatted usage report."""
        analysis = self.analyze_usage_patterns(days_back)

        if "error" in analysis:
            return f"📊 Agent Usage Report: {analysis['error']}"

        report = []
        report.append("📊 AGENT USAGE ANALYTICS REPORT")
        report.append("=" * 50)
        report.append(f"📅 Period: {analysis['date_range']}")
        report.append(f"🔢 Total Invocations: {analysis['total_invocations']}")
        report.append(f"✅ Success Rate: {analysis['success_rate']:.1%}")
        report.append(f"⏱️ Average Duration: {analysis['average_duration']:.1f}s")
        report.append("")

        # Agent usage breakdown
        report.append("🤖 AGENT USAGE FREQUENCY:")
        for agent, count in analysis["agent_usage"].most_common():
            percentage = count / analysis["total_invocations"] * 100
            report.append(f"   {agent}: {count} ({percentage:.1f}%)")
        report.append("")

        # Task types
        report.append("📋 TASK TYPES:")
        for task, count in analysis["task_types"].most_common(5):
            report.append(f"   {task}: {count}")
        report.append("")

        # Effectiveness by agent
        report.append("📈 AGENT EFFECTIVENESS:")
        for agent, stats in analysis["effectiveness_by_agent"].items():
            report.append(f"   {agent}:")
            report.append(f"     Success Rate: {stats['success_rate']:.1%}")
            report.append(f"     Avg Duration: {stats['avg_duration']:.1f}s")
            report.append(f"     Most Common: {stats['most_common_task'][0]}")
        report.append("")

        # Boundary violations
        if analysis["boundary_violations"]["total_violations"] > 0:
            report.append("⚠️ BOUNDARY VIOLATIONS:")
            for agent, violations in analysis["boundary_violations"]["violations_by_agent"].items():
                if violations:
                    report.append(f"   {agent}: {len(violations)} violations")
            report.append("")

        # Recommendations
        if analysis["optimization_recommendations"]:
            report.append("💡 OPTIMIZATION RECOMMENDATIONS:")
            report.extend(f"   {rec}" for rec in analysis["optimization_recommendations"])

        return "\n".join(report)

    def monitor_current_session(self, input_data: dict[str, Any]) -> AgentUsageMetric | None:
        """Monitor current tool usage for agent patterns."""
        try:
            # Extract agent usage from tool input
            user_prompt = input_data.get("user_prompt", "")

            # Simple agent detection from prompt
            agent_keywords = {
                "database-specialist": ["database", "query", "sql", "postgres", "schema"],
                "ml-orchestrator": ["model", "training", "ml", "machine learning", "feature"],
                "performance-engineer": ["performance", "optimize", "slow", "bottleneck", "monitoring"],
                "security-architect": ["security", "auth", "vulnerability", "encrypt", "secure"],
                "infrastructure-specialist": ["docker", "container", "deploy", "ci/cd", "testcontainer"]
            }

            detected_agent = None
            for agent, keywords in agent_keywords.items():
                if any(keyword in user_prompt.lower() for keyword in keywords):
                    detected_agent = agent
                    break

            if detected_agent:
                return AgentUsageMetric(
                    agent_name=detected_agent,
                    task_type="auto_detected",
                    timestamp=datetime.now(),
                    user_prompt=user_prompt[:200]  # Truncate for privacy
                )

        except Exception as e:
            print(f"Error monitoring session: {e}", file=sys.stderr)

        return None


def main():
    """Main function for standalone usage analysis."""
    tracker = AgentMetricsTracker()

    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "report":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            print(tracker.generate_report(days))

        elif command == "analyze":
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
            analysis = tracker.analyze_usage_patterns(days)
            print(json.dumps(analysis, indent=2, default=str))

        elif command == "monitor":
            # Monitor current tool input
            try:
                input_data = json.loads(sys.stdin.read())
                metric = tracker.monitor_current_session(input_data)
                if metric:
                    tracker.record_usage(metric)
                    print(f"Recorded usage for {metric.agent_name}")
            except Exception as e:
                print(f"Error monitoring: {e}", file=sys.stderr)
                sys.exit(1)

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    else:
        # Default: show recent report
        print(tracker.generate_report(7))


if __name__ == "__main__":
    main()
