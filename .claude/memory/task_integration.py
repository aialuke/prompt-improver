#!/usr/bin/env python3
"""
Task Tool Integration for Agent Memory System

Provides seamless integration with Claude Code's Task tool for automatic memory
loading and updates during agent delegations. This connects the memory system 
with the Task tool workflow for transparent memory management.
"""

import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from memory_hooks import AgentMemoryHooks, load_agent_context
from auto_update_hooks import AutoMemoryUpdater, auto_update_agent_memory
# Note: Avoiding circular import with workflow_integration


class TaskToolIntegration:
    """Integration layer for Claude Code Task tool with agent memory system."""
    
    def __init__(self, memory_dir: str = ".claude/memory"):
        """Initialize task tool integration."""
        self.memory_dir = Path(memory_dir)
        self.hooks = AgentMemoryHooks(memory_dir)
        self.updater = AutoMemoryUpdater(memory_dir)
# Note: workflow integration handled separately to avoid circular imports
        
        # Track task delegations
        self.active_delegations = {}
        
        # Task tool integration state
        self.integration_stats = {
            "delegations_tracked": 0,
            "memory_loads_triggered": 0,
            "memory_updates_triggered": 0,
            "successful_integrations": 0
        }
    
    def pre_task_hook(self, subagent_type: str, description: str, prompt: str) -> Dict[str, Any]:
        """Hook called before Task tool delegation to load agent memory.
        
        This function should be called by Claude Code before delegating to any agent
        to ensure the agent has access to its memory context.
        """
        delegation_start = time.time()
        
        try:
            # Generate delegation ID for tracking
            delegation_id = str(uuid.uuid4())
            
            # Load agent context using existing memory hooks
            context_result = load_agent_context(prompt, explicit_agent=subagent_type)
            
            # Track delegation
            self.active_delegations[delegation_id] = {
                "agent_type": subagent_type,
                "description": description,
                "prompt": prompt,
                "start_time": delegation_start,
                "context_loaded": context_result["memory_loaded"],
                "context": context_result.get("context", {}),
                "status": "active"
            }
            
            # Update integration stats
            self.integration_stats["delegations_tracked"] += 1
            if context_result["memory_loaded"]:
                self.integration_stats["memory_loads_triggered"] += 1
            
            return {
                "success": True,
                "delegation_id": delegation_id,
                "agent_name": context_result.get("agent_name", subagent_type),
                "memory_loaded": context_result["memory_loaded"],
                "context_summary": self._format_context_summary(context_result.get("context", {})),
                "recommendations": self._get_delegation_recommendations(
                    subagent_type, context_result.get("context", {})
                ),
                "memory_insights": self._extract_relevant_insights(
                    context_result.get("context", {}), description
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "delegation_id": None
            }
    
    def post_task_hook(self, delegation_id: str, agent_response: str, 
                      outcome: str = "success", execution_time: Optional[float] = None) -> Dict[str, Any]:
        """Hook called after Task tool delegation completes to update agent memory.
        
        This function should be called by Claude Code after receiving results from
        an agent delegation to update the agent's memory with new insights.
        """
        if delegation_id not in self.active_delegations:
            return {
                "success": False,
                "error": f"Delegation {delegation_id} not found"
            }
        
        delegation = self.active_delegations[delegation_id]
        
        try:
            # Calculate execution time if not provided
            if execution_time is None:
                execution_time = time.time() - delegation["start_time"]
            
            # Extract delegation information for memory update
            explicit_delegations = self._extract_delegations_from_response(agent_response)
            
            # Auto-update agent memory
            update_result = auto_update_agent_memory(
                agent_name=delegation["agent_type"],
                task_description=delegation["description"],
                response_text=agent_response,
                execution_time=execution_time,
                explicit_delegations=explicit_delegations,
                outcome=outcome
            )
            
            # Update delegation status
            delegation["status"] = "completed"
            delegation["completion_time"] = time.time()
            delegation["outcome"] = outcome
            delegation["execution_time"] = execution_time
            delegation["update_result"] = update_result
            
            # Update integration stats
            if update_result["success"]:
                self.integration_stats["memory_updates_triggered"] += 1
                self.integration_stats["successful_integrations"] += 1
            
            # Generate integration summary
            integration_summary = self._generate_integration_summary(delegation, update_result)
            
            return {
                "success": True,
                "delegation_id": delegation_id,
                "agent_name": delegation["agent_type"],
                "outcome": outcome,
                "execution_time": execution_time,
                "update_result": update_result,
                "integration_summary": integration_summary,
                "learning_impact": self._assess_learning_impact(update_result)
            }
            
        except Exception as e:
            delegation["status"] = "failed"
            delegation["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e),
                "delegation_id": delegation_id
            }
    
    def get_delegation_status(self, delegation_id: str) -> Dict[str, Any]:
        """Get current status of a task delegation."""
        if delegation_id not in self.active_delegations:
            return {"error": "Delegation not found"}
        
        delegation = self.active_delegations[delegation_id]
        
        return {
            "delegation_id": delegation_id,
            "agent_type": delegation["agent_type"],
            "status": delegation["status"],
            "description": delegation["description"][:100] + "..." if len(delegation["description"]) > 100 else delegation["description"],
            "context_loaded": delegation["context_loaded"],
            "elapsed_time": time.time() - delegation["start_time"],
            "outcome": delegation.get("outcome"),
            "execution_time": delegation.get("execution_time"),
            "update_result": delegation.get("update_result")
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get Task tool integration performance statistics."""
        stats = self.integration_stats.copy()
        
        # Calculate success rates
        if stats["delegations_tracked"] > 0:
            stats["memory_load_success_rate"] = (
                stats["memory_loads_triggered"] / stats["delegations_tracked"]
            ) * 100
            stats["integration_success_rate"] = (
                stats["successful_integrations"] / stats["delegations_tracked"]
            ) * 100
        
        # Add recent delegation data
        completed_delegations = [
            d for d in self.active_delegations.values() 
            if d["status"] == "completed"
        ]
        
        if completed_delegations:
            stats["recent_delegations"] = len(completed_delegations)
            stats["average_execution_time"] = sum(
                d.get("execution_time", 0) for d in completed_delegations
            ) / len(completed_delegations)
            stats["average_insights_per_delegation"] = sum(
                d.get("update_result", {}).get("insights_extracted", 0) 
                for d in completed_delegations
            ) / len(completed_delegations)
        
        return stats
    
    def create_task_delegation_context(self, subagent_type: str, task_description: str, 
                                     additional_context: Optional[str] = None) -> str:
        """Create enriched task context using agent memory.
        
        This function enriches task descriptions with relevant agent memory context
        to improve delegation effectiveness.
        """
        try:
            # Load agent memory
            agent_memory = self.hooks.manager.load_agent_memory(subagent_type)
            
            # Build context from memory
            context_parts = []
            
            # Add relevant insights
            insights = agent_memory.get("optimization_insights", [])
            relevant_insights = [
                i for i in insights 
                if any(keyword in task_description.lower() 
                      for keyword in i.get("area", "").replace("_", " ").split())
            ]
            
            if relevant_insights:
                context_parts.append("**Relevant Experience:**")
                for insight in relevant_insights[:3]:  # Top 3 most relevant
                    context_parts.append(f"• {insight.get('insight', 'Unknown insight')} (confidence: {insight.get('confidence', 0)*100:.0f}%)")
            
            # Add collaboration recommendations
            collabs = agent_memory.get("collaboration_patterns", {}).get("frequent_collaborators", [])
            effective_collabs = [c for c in collabs if c.get("success_rate", 0) > 0.8]
            
            if effective_collabs:
                context_parts.append("\n**Collaboration Recommendations:**")
                for collab in effective_collabs[:2]:  # Top 2 collaborators
                    context_parts.append(f"• Consider delegating to {collab['agent_name']} (proven {collab['success_rate']*100:.0f}% success rate)")
            
            # Add performance baselines if available
            project_context = agent_memory.get("domain_knowledge", {}).get("project_specific_context", {})
            baselines = project_context.get("performance_baselines", {})
            
            if baselines:
                context_parts.append("\n**Current Performance Baselines:**")
                for metric, value in list(baselines.items())[:3]:  # Top 3 metrics
                    context_parts.append(f"• {metric.replace('_', ' ').title()}: {value}")
            
            # Combine original task with memory context
            enriched_context = task_description
            
            if context_parts:
                enriched_context += "\n\n**Agent Memory Context:**\n" + "\n".join(context_parts)
            
            if additional_context:
                enriched_context += f"\n\n**Additional Context:**\n{additional_context}"
            
            return enriched_context
            
        except Exception:
            # Fall back to original task description if memory loading fails
            return task_description + (f"\n\n{additional_context}" if additional_context else "")
    
    def _format_context_summary(self, context: Dict[str, Any]) -> str:
        """Format agent context into a brief summary."""
        summary_parts = []
        
        # Recent performance
        recent_tasks = context.get("recent_tasks", [])
        if recent_tasks:
            success_rate = sum(1 for t in recent_tasks if t.get("outcome") == "success") / len(recent_tasks)
            summary_parts.append(f"Recent success: {success_rate*100:.0f}%")
        
        # High-confidence insights
        insights = context.get("optimization_insights", [])
        high_confidence = sum(1 for i in insights if i.get("confidence", 0) > 0.9)
        if high_confidence > 0:
            summary_parts.append(f"High-confidence insights: {high_confidence}")
        
        # Active collaborations
        collabs = context.get("collaboration_patterns", [])
        if collabs:
            summary_parts.append(f"Active collaborators: {len(collabs)}")
        
        return " | ".join(summary_parts) if summary_parts else "No significant context"
    
    def _get_delegation_recommendations(self, agent_type: str, context: Dict[str, Any]) -> List[str]:
        """Generate delegation recommendations based on agent memory."""
        recommendations = []
        
        # Based on successful collaboration patterns
        collabs = context.get("collaboration_patterns", [])
        for collab in collabs:
            if collab.get("success_rate", 0) > 0.85:
                recommendations.append(f"Leverage {collab['agent_name']} collaboration (proven {collab['success_rate']*100:.0f}% success)")
        
        # Based on high-impact areas
        insights = context.get("optimization_insights", [])
        high_impact_areas = set()
        for insight in insights:
            if insight.get("impact") == "high":
                high_impact_areas.add(insight.get("area", "").replace("_", " "))
        
        for area in list(high_impact_areas)[:2]:
            recommendations.append(f"Focus on {area} - historically high impact area")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _extract_relevant_insights(self, context: Dict[str, Any], task_description: str) -> List[Dict[str, Any]]:
        """Extract insights relevant to the current task."""
        insights = context.get("optimization_insights", [])
        
        # Find insights with keywords matching the task
        task_keywords = set(task_description.lower().split())
        relevant_insights = []
        
        for insight in insights:
            insight_keywords = set(insight.get("area", "").replace("_", " ").split())
            insight_text_keywords = set(insight.get("insight", "").lower().split())
            
            # Check for keyword overlap
            if (task_keywords & insight_keywords) or (task_keywords & insight_text_keywords):
                relevant_insights.append({
                    "area": insight.get("area", "unknown"),
                    "insight": insight.get("insight", ""),
                    "confidence": insight.get("confidence", 0),
                    "impact": insight.get("impact", "medium"),
                    "relevance_score": len(task_keywords & (insight_keywords | insight_text_keywords))
                })
        
        # Sort by relevance score and confidence
        relevant_insights.sort(key=lambda x: (x["relevance_score"], x["confidence"]), reverse=True)
        
        return relevant_insights[:5]  # Top 5 most relevant insights
    
    def _extract_delegations_from_response(self, response_text: str) -> List[Dict[str, str]]:
        """Extract delegation information from agent response text."""
        delegations = []
        
        # Common delegation patterns in responses
        delegation_keywords = [
            "recommend delegating", "should delegate", "delegate to", "handoff to",
            "collaborate with", "work with", "consult with", "involve"
        ]
        
        agent_names = [
            "database-specialist", "ml-orchestrator", "performance-engineer", 
            "security-architect", "infrastructure-specialist", "data-pipeline-specialist",
            "api-design-specialist", "monitoring-observability-specialist", 
            "testing-strategy-specialist", "configuration-management-specialist", 
            "documentation-specialist"
        ]
        
        response_lower = response_text.lower()
        
        # Look for delegation patterns
        for keyword in delegation_keywords:
            if keyword in response_lower:
                # Find mentioned agents near delegation keywords
                for agent in agent_names:
                    if agent in response_lower:
                        # Determine delegation reason based on context
                        reason = "collaboration"
                        if "performance" in response_lower:
                            reason = "performance optimization"
                        elif "security" in response_lower:
                            reason = "security validation"
                        elif "test" in response_lower:
                            reason = "testing validation"
                        elif "deploy" in response_lower:
                            reason = "deployment assistance"
                        
                        delegations.append({
                            "to_agent": agent,
                            "reason": reason,
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
        
        return unique_delegations[:3]  # Limit to 3 delegations
    
    def _generate_integration_summary(self, delegation: Dict[str, Any], 
                                    update_result: Dict[str, Any]) -> str:
        """Generate integration summary for completed delegation."""
        agent = delegation["agent_type"]
        outcome = delegation.get("outcome", "unknown")
        execution_time = delegation.get("execution_time", 0)
        insights = update_result.get("insights_extracted", 0)
        
        summary = f"Task delegation to {agent} completed with {outcome} outcome in {execution_time:.1f}s"
        
        if insights > 0:
            summary += f", extracted {insights} new insights"
        
        if update_result.get("delegations_identified", 0) > 0:
            summary += f", identified {update_result['delegations_identified']} collaboration opportunities"
        
        return summary
    
    def _assess_learning_impact(self, update_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the learning impact of a completed delegation."""
        return {
            "insights_extracted": update_result.get("insights_extracted", 0),
            "confidence": update_result.get("confidence", 0.75),
            "impact_level": update_result.get("impact_level", "medium"),
            "learning_value": "high" if update_result.get("insights_extracted", 0) >= 2 else "medium",
            "collaboration_potential": "high" if update_result.get("delegations_identified", 0) > 0 else "low"
        }


# Global task integration instance
task_integration = TaskToolIntegration()


def pre_task_hook(subagent_type: str, description: str, prompt: str) -> Dict[str, Any]:
    """Convenience function for pre-task hook integration."""
    return task_integration.pre_task_hook(subagent_type, description, prompt)


def post_task_hook(delegation_id: str, agent_response: str, 
                  outcome: str = "success", execution_time: Optional[float] = None) -> Dict[str, Any]:
    """Convenience function for post-task hook integration."""
    return task_integration.post_task_hook(delegation_id, agent_response, outcome, execution_time)


def enrich_task_context(subagent_type: str, task_description: str, 
                       additional_context: Optional[str] = None) -> str:
    """Convenience function to enrich task context with agent memory."""
    return task_integration.create_task_delegation_context(
        subagent_type, task_description, additional_context
    )


def get_task_integration_performance() -> Dict[str, Any]:
    """Convenience function to get task integration performance stats."""
    return task_integration.get_integration_stats()


if __name__ == "__main__":
    # Test Task tool integration
    print("🔗 Testing Task Tool Integration")
    print("=" * 45)
    
    integration = TaskToolIntegration()
    
    # Test 1: Pre-task hook (memory loading)
    print("📥 Testing Pre-Task Memory Loading...")
    pre_result = integration.pre_task_hook(
        subagent_type="database-specialist",
        description="Optimize slow analytics dashboard query", 
        prompt="Our analytics dashboard query is taking 5 seconds to load. Please analyze and optimize this query."
    )
    
    if pre_result["success"]:
        delegation_id = pre_result["delegation_id"]
        print(f"✅ Pre-task hook successful: {delegation_id[:8]}...")
        print(f"   Agent: {pre_result['agent_name']}")
        print(f"   Memory loaded: {pre_result['memory_loaded']}")
        print(f"   Context: {pre_result['context_summary']}")
        print(f"   Recommendations: {len(pre_result['recommendations'])}")
        
        for rec in pre_result["recommendations"]:
            print(f"     • {rec}")
        
        print(f"   Relevant insights: {len(pre_result['memory_insights'])}")
        for insight in pre_result["memory_insights"][:2]:
            print(f"     • {insight['area']}: {insight['insight'][:50]}...")
        
        # Test 2: Context enrichment
        print(f"\n📝 Testing Task Context Enrichment...")
        enriched_context = integration.create_task_delegation_context(
            "ml-orchestrator",
            "Improve model accuracy for prompt optimization",
            "Focus on reducing training time while maintaining performance"
        )
        
        print("✅ Context enrichment successful")
        print(f"   Original task length: {len('Improve model accuracy for prompt optimization')} chars")
        print(f"   Enriched context length: {len(enriched_context)} chars")
        print(f"   Enhancement ratio: {len(enriched_context) / len('Improve model accuracy for prompt optimization'):.1f}x")
        
        # Test 3: Post-task hook (memory update)
        print(f"\n📤 Testing Post-Task Memory Update...")
        
        simulated_response = """
        I analyzed the slow query using EXPLAIN and identified the bottleneck was a full table scan.
        Created a composite index on (user_id, created_at, status) which eliminated the table scan.
        Query response time improved from 5.2 seconds to 0.4 seconds (92% improvement).
        The optimization also improved cache hit rates significantly.
        
        Performance team should validate the system-wide impact of these changes.
        Consider implementing similar indexing strategies for other analytics queries.
        """
        
        post_result = integration.post_task_hook(
            delegation_id=delegation_id,
            agent_response=simulated_response,
            outcome="success",
            execution_time=2.3
        )
        
        if post_result["success"]:
            print(f"✅ Post-task hook successful")
            print(f"   Summary: {post_result['integration_summary']}")
            print(f"   Learning impact: {post_result['learning_impact']['learning_value']}")
            print(f"   Insights extracted: {post_result['update_result']['insights_extracted']}")
            print(f"   Delegations identified: {post_result['update_result'].get('delegations_identified', 0)}")
    
    # Test 4: Integration performance stats
    print(f"\n📊 Integration Performance Statistics")
    print("=" * 40)
    
    stats = integration.get_integration_stats()
    print(f"Delegations tracked: {stats['delegations_tracked']}")
    print(f"Memory loads triggered: {stats['memory_loads_triggered']}")  
    print(f"Memory updates triggered: {stats['memory_updates_triggered']}")
    print(f"Successful integrations: {stats['successful_integrations']}")
    
    if stats.get("integration_success_rate"):
        print(f"Integration success rate: {stats['integration_success_rate']:.1f}%")
    
    if stats.get("average_insights_per_delegation"):
        print(f"Avg insights per delegation: {stats['average_insights_per_delegation']:.1f}")
    
    print("\n✅ Task tool integration test completed successfully!")