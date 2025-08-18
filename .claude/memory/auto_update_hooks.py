#!/usr/bin/env python3
"""
Automatic Memory Update Hooks

Provides sophisticated automatic memory updates after task completion,
including intelligent insight extraction, performance impact analysis,
and collaboration pattern learning.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from memory_hooks import AgentMemoryHooks
from memory_manager import AgentMemoryManager


class AutoMemoryUpdater:
    """Automatic memory update system for post-task processing."""
    
    def __init__(self, memory_dir: str = ".claude/memory"):
        """Initialize auto memory updater."""
        self.hooks = AgentMemoryHooks(memory_dir)
        self.manager = AgentMemoryManager(memory_dir)
        
        # Performance keywords for insight extraction
        self.performance_keywords = {
            "high": ["dramatic", "significant", "major", "substantial", "outstanding", "excellent", 
                    "90%", "95%", "99%", "100%", "10x", "5x", "massive", "breakthrough"],
            "medium": ["improved", "better", "enhanced", "optimized", "faster", "efficient",
                      "50%", "60%", "70%", "80%", "2x", "3x", "good", "solid"],
            "low": ["slight", "minor", "small", "minimal", "basic", "10%", "20%", "30%",
                   "simple", "standard"]
        }
        
        # Domain-specific insight patterns
        self.domain_patterns = {
            "database-specialist": {
                "areas": ["query_optimization", "indexing", "connection_pooling", "schema_design", 
                         "migration", "performance_tuning"],
                "success_indicators": ["response time", "query speed", "index created", "cache hit",
                                     "connection pool", "schema optimized"],
                "delegation_triggers": ["system performance", "application speed", "bottleneck"]
            },
            "ml-orchestrator": {
                "areas": ["training_optimization", "model_performance", "feature_engineering", 
                         "deployment_strategy", "hyperparameter_tuning"],
                "success_indicators": ["accuracy improved", "training time", "model performance",
                                     "feature selection", "hyperparameter", "deployment"],
                "delegation_triggers": ["infrastructure scaling", "data pipeline", "performance validation"]
            },
            "performance-engineer": {
                "areas": ["system_optimization", "bottleneck_analysis", "caching", "monitoring",
                         "load_testing"],
                "success_indicators": ["response time", "throughput", "latency", "cache hit rate",
                                     "system load", "performance improvement"],
                "delegation_triggers": ["database optimization", "ml performance", "infrastructure tuning"]
            },
            "security-architect": {
                "areas": ["authentication", "authorization", "vulnerability_assessment", 
                         "encryption", "compliance"],
                "success_indicators": ["security improved", "vulnerability fixed", "authentication",
                                     "encryption", "compliance", "threat mitigation"],
                "delegation_triggers": ["security deployment", "infrastructure hardening"]
            },
            "infrastructure-specialist": {
                "areas": ["container_optimization", "deployment_strategy", "ci_cd", "testing_infrastructure",
                         "resource_management"],
                "success_indicators": ["deployment time", "container performance", "test reliability",
                                     "resource usage", "infrastructure"],
                "delegation_triggers": ["database infrastructure", "security tooling", "performance monitoring"]
            }
        }
    
    def extract_insights_from_response(self, response_text: str, agent_name: str) -> Dict[str, Any]:
        """Extract insights from task response text using NLP patterns."""
        insights = {
            "key_insights": [],
            "performance_improvements": [],
            "optimization_area": None,
            "optimization_insight": None,
            "impact_level": "medium",
            "confidence": 0.75,
            "delegations": [],
            "success_indicators": []
        }
        
        if not response_text or not agent_name:
            return insights
        
        response_lower = response_text.lower()
        
        # Extract performance improvements
        performance_patterns = [
            r"(\d+)%\s+improvement",
            r"improved.*by\s+(\d+)%",
            r"reduced.*from\s+([\d.]+).*to\s+([\d.]+)",
            r"increased.*from\s+([\d.]+).*to\s+([\d.]+)",
            r"response time.*(\d+)ms",
            r"(\d+)x\s+faster"
        ]
        
        for pattern in performance_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        insights["performance_improvements"].append(f"Performance change: {' -> '.join(match)}")
                    else:
                        insights["performance_improvements"].append(f"Performance improvement: {match}")
        
        # Determine impact level based on keywords
        for impact, keywords in self.performance_keywords.items():
            for keyword in keywords:
                if keyword in response_lower:
                    insights["impact_level"] = impact
                    break
            if insights["impact_level"] == impact:
                break
        
        # Extract domain-specific insights
        if agent_name in self.domain_patterns:
            domain = self.domain_patterns[agent_name]
            
            # Find optimization area
            for area in domain["areas"]:
                area_keywords = area.replace("_", " ").split()
                if any(keyword in response_lower for keyword in area_keywords):
                    insights["optimization_area"] = area
                    break
            
            # Extract success indicators
            for indicator in domain["success_indicators"]:
                if indicator in response_lower:
                    insights["success_indicators"].append(indicator)
            
            # Detect delegation opportunities
            for trigger in domain["delegation_triggers"]:
                if trigger in response_lower:
                    # Infer target agent based on trigger
                    target_agent = self.infer_delegation_target(trigger, agent_name)
                    if target_agent:
                        insights["delegations"].append({
                            "to_agent": target_agent,
                            "reason": trigger,
                            "outcome": "recommended"  # Mark as recommended
                        })
        
        # Extract key insights from sentences
        sentences = re.split(r'[.!?]+', response_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and any(keyword in sentence.lower() for keyword in 
                                        ["optimized", "improved", "created", "implemented", "reduced", "increased", "fixed"]):
                insights["key_insights"].append(sentence)
        
        # Limit insights to most relevant
        insights["key_insights"] = insights["key_insights"][:5]  # Top 5
        insights["performance_improvements"] = insights["performance_improvements"][:3]  # Top 3
        
        # Generate optimization insight
        if insights["optimization_area"] and insights["success_indicators"]:
            indicators = ", ".join(insights["success_indicators"][:2])
            insights["optimization_insight"] = f"Applied {insights['optimization_area'].replace('_', ' ')} techniques with {indicators}"
        
        # Set confidence based on evidence
        evidence_count = (len(insights["key_insights"]) + 
                         len(insights["performance_improvements"]) + 
                         len(insights["success_indicators"]))
        insights["confidence"] = min(0.95, 0.60 + (evidence_count * 0.05))
        
        return insights
    
    def detect_delegations_from_response(self, response_text: str) -> List[Dict[str, str]]:
        """Detect delegation recommendations from agent response text."""
        delegations = []
        
        if not response_text:
            return delegations
        
        response_lower = response_text.lower()
        
        # Common delegation patterns in responses
        delegation_keywords = [
            "recommend delegating", "should delegate", "delegate to", "handoff to",
            "collaborate with", "work with", "consult with", "involve",
            "should validate", "team should", "consider involving"
        ]
        
        # Known agent names for detection
        agent_names = [
            "database-specialist", "ml-orchestrator", "performance-engineer", 
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist", 
            "testing-strategy-specialist", "configuration-management-specialist", 
            "documentation-specialist"
        ]
        
        # Look for explicit delegation patterns
        for keyword in delegation_keywords:
            if keyword in response_lower:
                # Find mentioned agents near delegation keywords
                for agent in agent_names:
                    if agent in response_lower:
                        # Determine delegation reason based on context
                        reason = self._determine_delegation_reason(response_lower, agent)
                        
                        delegations.append({
                            "to_agent": agent,
                            "reason": reason,
                            "outcome": "recommended"
                        })
        
        # Look for implicit delegation patterns (contextual hints)
        implicit_patterns = {
            "performance": ["performance-engineer"],
            "security": ["security-architect"],
            "database": ["database-specialist"],
            "infrastructure": ["infrastructure-specialist"],
            "testing": ["testing-strategy-specialist"],
            "documentation": ["documentation-specialist"],
            "monitoring": ["monitoring-observability-specialist"],
            "deployment": ["infrastructure-specialist"],
            "ml model": ["ml-orchestrator"],
            "data pipeline": ["data-pipeline-specialist"]
        }
        
        # Check for validation/review requests
        validation_keywords = ["should validate", "needs validation", "validate", "review", "check"]
        for keyword in validation_keywords:
            if keyword in response_lower:
                for pattern, agents in implicit_patterns.items():
                    if pattern in response_lower:
                        for agent in agents:
                            delegations.append({
                                "to_agent": agent,
                                "reason": f"{pattern} validation",
                                "outcome": "recommended"
                            })
        
        # Remove duplicates
        seen = set()
        unique_delegations = []
        for delegation in delegations:
            key = (delegation["to_agent"], delegation["reason"])
            if key not in seen:
                seen.add(key)
                unique_delegations.append(delegation)
        
        return unique_delegations[:3]  # Limit to top 3 most relevant
    
    def _determine_delegation_reason(self, response_text: str, target_agent: str) -> str:
        """Determine the reason for delegation based on context and target agent."""
        # Agent-specific reason mapping
        agent_reasons = {
            "performance-engineer": "performance optimization",
            "database-specialist": "database optimization", 
            "security-architect": "security validation",
            "infrastructure-specialist": "infrastructure scaling",
            "ml-orchestrator": "model optimization",
            "testing-strategy-specialist": "testing validation",
            "documentation-specialist": "documentation update",
            "monitoring-observability-specialist": "monitoring setup",
            "data-pipeline-specialist": "data pipeline optimization",
            "api-design-specialist": "api design review",
            "configuration-management-specialist": "configuration management"
        }
        
        # Context-based reason detection
        context_keywords = {
            "performance": "performance optimization",
            "security": "security validation",
            "test": "testing validation", 
            "deploy": "deployment assistance",
            "monitor": "monitoring setup",
            "document": "documentation update",
            "config": "configuration management",
            "scale": "infrastructure scaling"
        }
        
        # Check for context-specific reasons
        for keyword, reason in context_keywords.items():
            if keyword in response_text:
                return reason
        
        # Fall back to agent-specific default
        return agent_reasons.get(target_agent, "collaboration")
    
    def infer_delegation_target(self, trigger: str, current_agent: str) -> Optional[str]:
        """Infer delegation target based on trigger and current agent."""
        delegation_map = {
            "system performance": "performance-engineer",
            "application speed": "performance-engineer", 
            "bottleneck": "performance-engineer",
            "performance validation": "performance-engineer",
            "database optimization": "database-specialist",
            "query optimization": "database-specialist",
            "ml performance": "ml-orchestrator",
            "infrastructure scaling": "infrastructure-specialist",
            "data pipeline": "data-pipeline-specialist",
            "security deployment": "infrastructure-specialist",
            "infrastructure hardening": "infrastructure-specialist",
            "security tooling": "infrastructure-specialist",
            "performance monitoring": "monitoring-observability-specialist",
            "feature engineering": "data-pipeline-specialist"
        }
        
        target = delegation_map.get(trigger)
        return target if target != current_agent else None
    
    def analyze_task_outcome(self, task_description: str, response_text: str, 
                           execution_time: Optional[float] = None) -> str:
        """Analyze task outcome based on description, response, and execution time."""
        if not response_text:
            return "failure"
        
        response_lower = response_text.lower()
        
        # Failure indicators
        failure_indicators = ["error", "failed", "exception", "couldn't", "unable", "broken", "issue"]
        if any(indicator in response_lower for indicator in failure_indicators):
            return "failure"
        
        # Partial success indicators
        partial_indicators = ["partially", "some issues", "limited", "temporary", "workaround"]
        if any(indicator in response_lower for indicator in partial_indicators):
            return "partial"
        
        # Success indicators
        success_indicators = ["completed", "successful", "optimized", "improved", "fixed", "implemented", "resolved"]
        if any(indicator in response_lower for indicator in success_indicators):
            return "success"
        
        # Default based on response length and execution time
        if len(response_text) > 100:  # Substantial response
            return "success"
        elif execution_time and execution_time > 30:  # Long execution suggests effort
            return "partial"
        
        return "partial"  # Default conservative outcome
    
    def auto_update_memory(self, agent_name: str, task_description: str, 
                          response_text: str, execution_time: Optional[float] = None,
                          explicit_delegations: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Automatically update agent memory from task completion."""
        try:
            # Extract insights from response
            insights = self.extract_insights_from_response(response_text, agent_name)
            
            # Analyze task outcome
            outcome = self.analyze_task_outcome(task_description, response_text, execution_time)
            
            # Combine explicit and inferred delegations
            all_delegations = []
            if explicit_delegations:
                all_delegations.extend(explicit_delegations)
            if insights["delegations"]:
                # Only add inferred delegations that aren't already explicit
                explicit_targets = {d.get("to_agent") for d in explicit_delegations or []}
                for delegation in insights["delegations"]:
                    if delegation["to_agent"] not in explicit_targets:
                        all_delegations.append(delegation)
            
            # Prepare task info
            task_info = {
                "task_id": str(uuid.uuid4()),
                "task_description": task_description[:200] + "..." if len(task_description) > 200 else task_description,
                "outcome": outcome,
                "key_insights": insights["key_insights"] + insights["performance_improvements"],
                "delegations": all_delegations,
                "duration_seconds": execution_time,
                "success_indicators": insights["success_indicators"]
            }
            
            # Update task history
            self.manager.add_task_to_history(agent_name, task_info)
            
            # Add optimization insight if found
            if insights["optimization_area"] and insights["optimization_insight"]:
                optimization_info = {
                    "area": insights["optimization_area"],
                    "insight": insights["optimization_insight"],
                    "impact": insights["impact_level"],
                    "confidence": insights["confidence"]
                }
                self.manager.add_optimization_insight(agent_name, optimization_info)
            
            # Update collaboration patterns
            for delegation in all_delegations:
                if delegation.get("outcome") in ["success", "partial"]:
                    success = delegation["outcome"] == "success"
                    task_type = delegation.get("reason", "unknown")
                    collaborator = delegation["to_agent"]
                    
                    self.manager.update_collaboration_pattern(
                        agent_name, collaborator, success, task_type
                    )
            
            # Send cross-agent insights if significant
            if insights["impact_level"] == "high" and insights["optimization_insight"]:
                message_id = self.manager.add_inter_agent_message(
                    from_agent=agent_name,
                    message_type="insight",
                    content=f"High-impact optimization: {insights['optimization_insight']}",
                    target_agents=[],  # Broadcast to all
                    metadata={
                        "priority": "high",
                        "impact_level": insights["impact_level"],
                        "confidence": insights["confidence"],
                        "task_id": task_info["task_id"]
                    }
                )
            
            return {
                "success": True,
                "task_id": task_info["task_id"],
                "outcome": outcome,
                "insights_extracted": len(insights["key_insights"]) + len(insights["performance_improvements"]),
                "optimization_area": insights["optimization_area"],
                "impact_level": insights["impact_level"],
                "confidence": insights["confidence"],
                "delegations_identified": len(all_delegations),
                "cross_agent_message": "sent" if insights["impact_level"] == "high" else "none"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "task_id": None
            }
    
    def batch_update_memories(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch update multiple agent memories."""
        results = {
            "successful_updates": 0,
            "failed_updates": 0,
            "total_insights": 0,
            "high_impact_insights": 0,
            "cross_agent_messages": 0,
            "results": []
        }
        
        for update in updates:
            result = self.auto_update_memory(
                agent_name=update["agent_name"],
                task_description=update["task_description"],
                response_text=update["response_text"],
                execution_time=update.get("execution_time"),
                explicit_delegations=update.get("delegations")
            )
            
            results["results"].append(result)
            
            if result["success"]:
                results["successful_updates"] += 1
                results["total_insights"] += result.get("insights_extracted", 0)
                if result.get("impact_level") == "high":
                    results["high_impact_insights"] += 1
                if result.get("cross_agent_message") == "sent":
                    results["cross_agent_messages"] += 1
            else:
                results["failed_updates"] += 1
        
        return results
    
    def get_memory_update_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get summary of recent memory updates for an agent."""
        try:
            memory = self.manager.load_agent_memory(agent_name)
            
            recent_tasks = memory["task_history"][:10]  # Last 10 tasks
            recent_insights = memory["optimization_insights"][-5:]  # Last 5 insights
            
            summary = {
                "agent_name": agent_name,
                "total_tasks": len(memory["task_history"]),
                "recent_task_outcomes": {},
                "recent_optimization_areas": {},
                "collaboration_effectiveness": {},
                "learning_trends": []
            }
            
            # Analyze recent task outcomes
            for task in recent_tasks:
                outcome = task["outcome"]
                summary["recent_task_outcomes"][outcome] = summary["recent_task_outcomes"].get(outcome, 0) + 1
            
            # Analyze optimization areas
            for insight in recent_insights:
                area = insight["area"]
                summary["recent_optimization_areas"][area] = summary["recent_optimization_areas"].get(area, 0) + 1
            
            # Analyze collaboration effectiveness
            for collab in memory["collaboration_patterns"]["frequent_collaborators"]:
                agent = collab["agent_name"]
                summary["collaboration_effectiveness"][agent] = {
                    "success_rate": collab["success_rate"],
                    "frequency": collab["collaboration_frequency"],
                    "common_tasks": collab.get("common_tasks", [])
                }
            
            # Identify learning trends
            if len(recent_tasks) >= 3:
                success_rate = sum(1 for t in recent_tasks if t["outcome"] == "success") / len(recent_tasks)
                if success_rate > 0.8:
                    summary["learning_trends"].append("High task success rate (>80%)")
                
                if len(recent_insights) >= 2:
                    high_confidence = sum(1 for i in recent_insights if i["confidence"] > 0.9)
                    if high_confidence >= 2:
                        summary["learning_trends"].append("High confidence optimization insights")
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}


# Global auto updater instance
auto_updater = AutoMemoryUpdater()


def auto_update_agent_memory(agent_name: str, task_description: str, response_text: str,
                            execution_time: Optional[float] = None,
                            explicit_delegations: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Convenience function for automatic memory updates."""
    return auto_updater.auto_update_memory(
        agent_name, task_description, response_text, execution_time, explicit_delegations
    )


def get_agent_memory_summary(agent_name: str) -> Dict[str, Any]:
    """Convenience function to get agent memory summary."""
    return auto_updater.get_memory_update_summary(agent_name)


if __name__ == "__main__":
    # Test automatic memory updates
    updater = AutoMemoryUpdater()
    
    # Test case 1: Database optimization
    print("🧪 Testing Automatic Memory Updates")
    print("=" * 50)
    
    db_response = """
    I optimized the slow PostgreSQL query by analyzing the EXPLAIN plan and identifying the bottleneck.
    Created a composite index on (user_id, created_at, status) which eliminated the table scan.
    Query response time improved from 5.2 seconds to 0.4 seconds - a 92% improvement.
    The optimization also improved cache hit rates and reduced database load significantly.
    System performance should be validated to ensure no regressions.
    """
    
    result1 = updater.auto_update_memory(
        agent_name="database-specialist",
        task_description="Optimize slow analytics dashboard query",
        response_text=db_response,
        execution_time=45.2
    )
    
    print("Database Optimization Result:")
    print(f"  Success: {result1['success']}")
    print(f"  Outcome: {result1['outcome']}")
    print(f"  Insights extracted: {result1['insights_extracted']}")
    print(f"  Impact level: {result1['impact_level']}")
    print(f"  Confidence: {result1['confidence']:.2f}")
    print(f"  Delegations: {result1['delegations_identified']}")
    
    # Test case 2: ML training
    ml_response = """
    ML model training optimization completed successfully. Hyperparameter tuning improved model accuracy 
    from 84% to 91% using grid search with cross-validation. Feature engineering with context-aware 
    learning patterns enhanced performance. Training time reduced by 35% through optimized batch processing.
    Infrastructure scaling may be needed for production deployment.
    """
    
    result2 = updater.auto_update_memory(
        agent_name="ml-orchestrator", 
        task_description="Optimize ML model training pipeline",
        response_text=ml_response,
        execution_time=120.5
    )
    
    print("\nML Training Result:")
    print(f"  Success: {result2['success']}")
    print(f"  Outcome: {result2['outcome']}")
    print(f"  Insights extracted: {result2['insights_extracted']}")
    print(f"  Optimization area: {result2['optimization_area']}")
    print(f"  Impact level: {result2['impact_level']}")
    
    # Get memory summaries
    print("\n📊 Memory Summaries:")
    print("=" * 30)
    
    for agent in ["database-specialist", "ml-orchestrator"]:
        summary = updater.get_memory_update_summary(agent)
        print(f"\n{agent}:")
        print(f"  Total tasks: {summary.get('total_tasks', 0)}")
        print(f"  Recent outcomes: {summary.get('recent_task_outcomes', {})}")
        print(f"  Learning trends: {summary.get('learning_trends', [])}")
    
    print("\n✅ Automatic memory update tests completed!")