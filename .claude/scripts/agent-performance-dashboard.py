#!/usr/bin/env python3
"""Agent Performance Dashboard - Real-time Agent Analytics

This script provides real-time analytics and optimization recommendations for Claude Code agents by:
1. Displaying agent usage trends and effectiveness metrics
2. Providing role boundary compliance monitoring
3. Suggesting optimization strategies based on usage patterns
4. Tracking delegation patterns and workflow efficiency
5. Monitoring agent ecosystem health and balance

Usage: python agent-performance-dashboard.py [--days N] [--format json|text]
"""

import json
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict

import sys
sys.path.append('.')
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agent_metrics_tracker import AgentMetricsTracker, AgentUsageMetric


class AgentPerformanceDashboard:
    """Advanced analytics dashboard for agent performance optimization."""
    
    def __init__(self, metrics_tracker: AgentMetricsTracker):
        self.tracker = metrics_tracker
        
        # Performance benchmarks (based on project standards)
        self.benchmarks = {
            "max_response_time": 300,  # 5 minutes
            "min_success_rate": 0.85,  # 85% success rate
            "max_violation_rate": 0.05,  # 5% boundary violations
            "delegation_efficiency": 0.7,  # 70% proper delegation
            "usage_balance_ratio": 3.0,  # Max 3:1 usage ratio between agents
        }

    def generate_performance_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance analysis."""
        metrics = self.tracker.load_metrics(days_back)
        
        if not metrics:
            return {"status": "no_data", "message": "No metrics available for analysis"}
        
        return {
            "metadata": {
                "report_generated": datetime.now().isoformat(),
                "analysis_period": f"{days_back} days",
                "total_metrics": len(metrics),
                "date_range": {
                    "start": min(m.timestamp for m in metrics).isoformat(),
                    "end": max(m.timestamp for m in metrics).isoformat()
                }
            },
            "agent_ecosystem_health": self._analyze_ecosystem_health(metrics),
            "individual_agent_performance": self._analyze_individual_performance(metrics),
            "delegation_workflow_analysis": self._analyze_delegation_workflows(metrics),
            "optimization_opportunities": self._identify_optimization_opportunities(metrics),
            "quality_trends": self._analyze_quality_trends(metrics),
            "usage_patterns": self._analyze_usage_patterns(metrics),
            "performance_benchmarks": self._benchmark_against_standards(metrics)
        }

    def _analyze_ecosystem_health(self, metrics: List[AgentUsageMetric]) -> Dict[str, Any]:
        """Analyze overall agent ecosystem health."""
        total_invocations = len(metrics)
        successful_invocations = sum(1 for m in metrics if m.success)
        
        # Calculate ecosystem metrics
        agent_counts = Counter(m.agent_name for m in metrics)
        task_diversity = len(set(m.task_type for m in metrics))
        
        # Delegation analysis
        delegations = [m for m in metrics if m.delegation_pattern]
        delegation_rate = len(delegations) / total_invocations if total_invocations > 0 else 0
        
        # Boundary violations
        violations = sum(len(m.boundary_violations) for m in metrics)
        violation_rate = violations / total_invocations if total_invocations > 0 else 0
        
        # Health score calculation (0-100)
        success_score = (successful_invocations / total_invocations) * 30 if total_invocations > 0 else 0
        balance_score = min(30, (min(agent_counts.values()) / max(agent_counts.values())) * 30) if agent_counts else 0
        delegation_score = min(20, delegation_rate * 100)
        violation_score = max(0, 20 - (violation_rate * 400))  # Penalty for violations
        
        health_score = success_score + balance_score + delegation_score + violation_score
        
        return {
            "overall_health_score": round(health_score, 1),
            "health_rating": self._get_health_rating(health_score),
            "ecosystem_metrics": {
                "total_invocations": total_invocations,
                "success_rate": successful_invocations / total_invocations if total_invocations > 0 else 0,
                "task_diversity": task_diversity,
                "delegation_rate": delegation_rate,
                "boundary_violation_rate": violation_rate
            },
            "agent_balance": {
                "most_used": max(agent_counts, key=agent_counts.get) if agent_counts else None,
                "least_used": min(agent_counts, key=agent_counts.get) if agent_counts else None,
                "usage_ratio": max(agent_counts.values()) / min(agent_counts.values()) if agent_counts and min(agent_counts.values()) > 0 else float('inf')
            }
        }

    def _analyze_individual_performance(self, metrics: List[AgentUsageMetric]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance of individual agents."""
        performance = defaultdict(lambda: {
            "total_invocations": 0,
            "successful_invocations": 0,
            "total_duration": 0,
            "task_types": Counter(),
            "quality_scores": Counter(),
            "boundary_violations": 0,
            "delegations_made": 0,
            "delegations_received": 0
        })
        
        # Collect metrics by agent
        for metric in metrics:
            agent_perf = performance[metric.agent_name]
            agent_perf["total_invocations"] += 1
            
            if metric.success:
                agent_perf["successful_invocations"] += 1
            
            if metric.duration_seconds:
                agent_perf["total_duration"] += metric.duration_seconds
            
            agent_perf["task_types"][metric.task_type] += 1
            agent_perf["quality_scores"][metric.outcome_quality] += 1
            agent_perf["boundary_violations"] += len(metric.boundary_violations)
            agent_perf["delegations_made"] += len(metric.delegation_pattern)
        
        # Calculate derived metrics
        for agent, perf in performance.items():
            total = perf["total_invocations"]
            if total > 0:
                perf["success_rate"] = perf["successful_invocations"] / total
                perf["avg_duration"] = perf["total_duration"] / total
                perf["violation_rate"] = perf["boundary_violations"] / total
                perf["delegation_frequency"] = perf["delegations_made"] / total
                
                # Performance score (0-100)
                success_component = perf["success_rate"] * 40
                speed_component = min(30, max(0, 30 - (perf["avg_duration"] / 10)))
                quality_component = (perf["quality_scores"].get("good", 0) / total) * 20 if total > 0 else 0
                compliance_component = max(0, 10 - (perf["violation_rate"] * 200))
                
                perf["performance_score"] = round(success_component + speed_component + quality_component + compliance_component, 1)
                perf["performance_rating"] = self._get_performance_rating(perf["performance_score"])
                
                # Most common task
                perf["primary_task"] = perf["task_types"].most_common(1)[0] if perf["task_types"] else ("none", 0)
        
        return dict(performance)

    def _analyze_delegation_workflows(self, metrics: List[AgentUsageMetric]) -> Dict[str, Any]:
        """Analyze delegation patterns and workflow efficiency."""
        delegations = [m for m in metrics if m.delegation_pattern]
        
        if not delegations:
            return {"status": "no_delegations", "message": "No delegation data available"}
        
        # Delegation network analysis
        delegation_network = defaultdict(Counter)
        for metric in delegations:
            for target in metric.delegation_pattern:
                delegation_network[metric.agent_name][target] += 1
        
        # Calculate delegation efficiency
        successful_delegations = sum(1 for m in delegations if m.success)
        delegation_success_rate = successful_delegations / len(delegations)
        
        # Common delegation paths
        common_paths = []
        for source, targets in delegation_network.items():
            for target, count in targets.most_common(3):
                common_paths.append({
                    "from": source,
                    "to": target,
                    "frequency": count,
                    "path": f"{source} → {target}"
                })
        
        return {
            "total_delegations": len(delegations),
            "delegation_success_rate": delegation_success_rate,
            "unique_delegation_paths": len([(s, t) for s, targets in delegation_network.items() for t in targets]),
            "most_common_paths": sorted(common_paths, key=lambda x: x["frequency"], reverse=True)[:5],
            "delegation_network": dict(delegation_network),
            "workflow_efficiency": delegation_success_rate * (len(delegations) / len(metrics)) if metrics else 0
        }

    def _identify_optimization_opportunities(self, metrics: List[AgentUsageMetric]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Analyze agent performance against benchmarks
        individual_perf = self._analyze_individual_performance(metrics)
        
        for agent, perf in individual_perf.items():
            # Low success rate
            if perf["success_rate"] < self.benchmarks["min_success_rate"]:
                opportunities.append({
                    "type": "low_success_rate",
                    "agent": agent,
                    "current_value": perf["success_rate"],
                    "benchmark": self.benchmarks["min_success_rate"],
                    "severity": "high" if perf["success_rate"] < 0.7 else "medium",
                    "recommendation": f"Review {agent} role definition and provide additional guidance. Current success rate: {perf['success_rate']:.1%}"
                })
            
            # Slow response times
            if perf["avg_duration"] > self.benchmarks["max_response_time"]:
                opportunities.append({
                    "type": "slow_response",
                    "agent": agent,
                    "current_value": perf["avg_duration"],
                    "benchmark": self.benchmarks["max_response_time"],
                    "severity": "medium",
                    "recommendation": f"Optimize {agent} configuration. Average duration: {perf['avg_duration']:.1f}s exceeds {self.benchmarks['max_response_time']}s target"
                })
            
            # High boundary violations
            if perf["violation_rate"] > self.benchmarks["max_violation_rate"]:
                opportunities.append({
                    "type": "boundary_violations",
                    "agent": agent,
                    "current_value": perf["violation_rate"],
                    "benchmark": self.benchmarks["max_violation_rate"],
                    "severity": "high",
                    "recommendation": f"Clarify {agent} role boundaries. Violation rate: {perf['violation_rate']:.1%} exceeds {self.benchmarks['max_violation_rate']:.1%} threshold"
                })
        
        # Usage imbalance
        agent_counts = Counter(m.agent_name for m in metrics)
        if agent_counts:
            max_usage = max(agent_counts.values())
            min_usage = min(agent_counts.values())
            
            if min_usage > 0 and max_usage / min_usage > self.benchmarks["usage_balance_ratio"]:
                overused = max(agent_counts, key=agent_counts.get)
                underused = min(agent_counts, key=agent_counts.get)
                
                opportunities.append({
                    "type": "usage_imbalance",
                    "agents": [overused, underused],
                    "current_ratio": max_usage / min_usage,
                    "benchmark": self.benchmarks["usage_balance_ratio"],
                    "severity": "medium",
                    "recommendation": f"Rebalance agent usage. {overused} is overused ({max_usage}) vs {underused} ({min_usage})"
                })
        
        return sorted(opportunities, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["severity"]], reverse=True)

    def _analyze_quality_trends(self, metrics: List[AgentUsageMetric]) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        # Group metrics by day
        daily_metrics = defaultdict(list)
        for metric in metrics:
            day = metric.timestamp.date()
            daily_metrics[day].append(metric)
        
        # Calculate daily quality scores
        daily_quality = {}
        for day, day_metrics in daily_metrics.items():
            total = len(day_metrics)
            successful = sum(1 for m in day_metrics if m.success)
            good_quality = sum(1 for m in day_metrics if m.outcome_quality == "good")
            
            quality_score = (successful / total * 0.7 + good_quality / total * 0.3) if total > 0 else 0
            daily_quality[day.isoformat()] = {
                "quality_score": quality_score,
                "total_tasks": total,
                "success_rate": successful / total if total > 0 else 0
            }
        
        # Calculate trend
        if len(daily_quality) >= 2:
            scores = [data["quality_score"] for data in daily_quality.values()]
            trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "daily_quality_scores": daily_quality,
            "overall_trend": trend,
            "average_quality": sum(data["quality_score"] for data in daily_quality.values()) / len(daily_quality) if daily_quality else 0
        }

    def _analyze_usage_patterns(self, metrics: List[AgentUsageMetric]) -> Dict[str, Any]:
        """Analyze usage patterns and temporal distribution."""
        # Hour of day analysis
        hourly_usage = Counter(m.timestamp.hour for m in metrics)
        
        # Day of week analysis
        daily_usage = Counter(m.timestamp.weekday() for m in metrics)
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Task type patterns
        task_patterns = Counter(m.task_type for m in metrics)
        
        return {
            "temporal_patterns": {
                "busiest_hour": max(hourly_usage, key=hourly_usage.get) if hourly_usage else None,
                "busiest_day": day_names[max(daily_usage, key=daily_usage.get)] if daily_usage else None,
                "hourly_distribution": dict(hourly_usage),
                "daily_distribution": {day_names[k]: v for k, v in daily_usage.items()}
            },
            "task_patterns": {
                "most_common_task": max(task_patterns, key=task_patterns.get) if task_patterns else None,
                "task_distribution": dict(task_patterns.most_common(10))
            }
        }

    def _benchmark_against_standards(self, metrics: List[AgentUsageMetric]) -> Dict[str, Any]:
        """Benchmark current performance against project standards."""
        total_metrics = len(metrics)
        if total_metrics == 0:
            return {"status": "no_data"}
        
        # Calculate current metrics
        success_rate = sum(1 for m in metrics if m.success) / total_metrics
        avg_duration = sum(m.duration_seconds or 0 for m in metrics) / total_metrics
        violation_rate = sum(len(m.boundary_violations) for m in metrics) / total_metrics
        
        benchmarks = {
            "success_rate": {
                "current": success_rate,
                "benchmark": self.benchmarks["min_success_rate"],
                "status": "pass" if success_rate >= self.benchmarks["min_success_rate"] else "fail",
                "gap": success_rate - self.benchmarks["min_success_rate"]
            },
            "response_time": {
                "current": avg_duration,
                "benchmark": self.benchmarks["max_response_time"],
                "status": "pass" if avg_duration <= self.benchmarks["max_response_time"] else "fail",
                "gap": self.benchmarks["max_response_time"] - avg_duration
            },
            "violation_rate": {
                "current": violation_rate,
                "benchmark": self.benchmarks["max_violation_rate"],
                "status": "pass" if violation_rate <= self.benchmarks["max_violation_rate"] else "fail",
                "gap": self.benchmarks["max_violation_rate"] - violation_rate
            }
        }
        
        passing_benchmarks = sum(1 for b in benchmarks.values() if b["status"] == "pass")
        overall_compliance = passing_benchmarks / len(benchmarks)
        
        return {
            "overall_compliance": overall_compliance,
            "compliance_rating": self._get_compliance_rating(overall_compliance),
            "individual_benchmarks": benchmarks,
            "passing_benchmarks": passing_benchmarks,
            "total_benchmarks": len(benchmarks)
        }

    def _get_health_rating(self, score: float) -> str:
        """Convert health score to rating."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        else:
            return "poor"

    def _get_performance_rating(self, score: float) -> str:
        """Convert performance score to rating."""
        if score >= 85:
            return "outstanding"
        elif score >= 70:
            return "good"
        elif score >= 55:
            return "acceptable"
        else:
            return "needs_improvement"

    def _get_compliance_rating(self, rate: float) -> str:
        """Convert compliance rate to rating."""
        if rate >= 0.9:
            return "fully_compliant"
        elif rate >= 0.7:
            return "mostly_compliant"
        elif rate >= 0.5:
            return "partially_compliant"
        else:
            return "non_compliant"

    def format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        if report.get("status") == "no_data":
            return "📊 No agent usage data available for analysis."
        
        lines = []
        lines.append("🚀 AGENT PERFORMANCE DASHBOARD")
        lines.append("=" * 50)
        
        metadata = report["metadata"]
        lines.append(f"📅 Analysis Period: {metadata['analysis_period']}")
        lines.append(f"📊 Total Metrics: {metadata['total_metrics']}")
        lines.append("")
        
        # Ecosystem health
        health = report["agent_ecosystem_health"]
        lines.append("🏥 ECOSYSTEM HEALTH")
        lines.append("-" * 20)
        lines.append(f"Overall Health: {health['overall_health_score']}/100 ({health['health_rating'].upper()})")
        lines.append(f"Success Rate: {health['ecosystem_metrics']['success_rate']:.1%}")
        lines.append(f"Delegation Rate: {health['ecosystem_metrics']['delegation_rate']:.1%}")
        lines.append(f"Violation Rate: {health['ecosystem_metrics']['boundary_violation_rate']:.1%}")
        lines.append("")
        
        # Individual performance
        individual = report["individual_agent_performance"]
        lines.append("🤖 AGENT PERFORMANCE")
        lines.append("-" * 20)
        for agent, perf in individual.items():
            lines.append(f"{agent}:")
            lines.append(f"  Score: {perf['performance_score']}/100 ({perf['performance_rating']})")
            lines.append(f"  Success: {perf['success_rate']:.1%}")
            lines.append(f"  Avg Duration: {perf['avg_duration']:.1f}s")
            lines.append(f"  Primary Task: {perf['primary_task'][0]}")
        lines.append("")
        
        # Optimization opportunities
        opportunities = report["optimization_opportunities"]
        if opportunities:
            lines.append("🎯 OPTIMIZATION OPPORTUNITIES")
            lines.append("-" * 30)
            for opp in opportunities[:5]:  # Show top 5
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[opp["severity"]]
                lines.append(f"{severity_icon} {opp['recommendation']}")
            lines.append("")
        
        # Benchmarks
        benchmarks = report["performance_benchmarks"]
        if benchmarks.get("status") != "no_data":
            lines.append("📏 BENCHMARK COMPLIANCE")
            lines.append("-" * 25)
            lines.append(f"Overall: {benchmarks['overall_compliance']:.1%} ({benchmarks['compliance_rating']})")
            for name, bench in benchmarks["individual_benchmarks"].items():
                status_icon = "✅" if bench["status"] == "pass" else "❌"
                lines.append(f"{status_icon} {name}: {bench['current']:.3f} (target: {bench['benchmark']:.3f})")
        
        return "\n".join(lines)


def main():
    """Main function for dashboard execution."""
    parser = argparse.ArgumentParser(description="Agent Performance Dashboard")
    parser.add_argument("--days", type=int, default=7, help="Days of data to analyze")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    tracker = AgentMetricsTracker()
    dashboard = AgentPerformanceDashboard(tracker)
    
    report = dashboard.generate_performance_report(args.days)
    
    if args.format == "json":
        output = json.dumps(report, indent=2, default=str)
    else:
        output = dashboard.format_text_report(report)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()