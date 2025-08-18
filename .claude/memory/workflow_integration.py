#!/usr/bin/env python3
"""
Complete Workflow Integration for Agent Memory System

Provides end-to-end integration combining memory loading hooks and automatic
memory updates for seamless agent memory management in Claude Code workflows.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from memory_hooks import AgentMemoryHooks, load_agent_context
from auto_update_hooks import AutoMemoryUpdater, auto_update_agent_memory
from task_integration import pre_task_hook, post_task_hook


class AgentMemoryWorkflow:
    """Complete workflow integration for agent memory system."""
    
    def __init__(self, memory_dir: str = ".claude/memory"):
        """Initialize workflow integration."""
        self.hooks = AgentMemoryHooks(memory_dir)
        self.updater = AutoMemoryUpdater(memory_dir)
        self.memory_dir = Path(memory_dir)
        
        # Track active workflows
        self.active_workflows = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "average_response_time": 0.0,
            "memory_operations": 0,
            "insights_generated": 0
        }
    
    def start_agent_workflow(self, task_prompt: str, explicit_agent: Optional[str] = None,
                           context_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start a complete agent workflow with memory loading."""
        workflow_start = time.time()
        
        try:
            # Load agent context
            context_result = load_agent_context(task_prompt, explicit_agent)
            
            if not context_result["memory_loaded"]:
                return {
                    "success": False,
                    "error": "No agent detected or memory loading failed",
                    "workflow_id": None
                }
            
            agent_name = context_result["agent_name"] 
            workflow_id = context_result["task_id"]
            
            # Store workflow state
            self.active_workflows[workflow_id] = {
                "agent_name": agent_name,
                "task_prompt": task_prompt,
                "start_time": workflow_start,
                "context": context_result["context"],
                "status": "active"
            }
            
            # Call context callback if provided
            if context_callback:
                context_callback(context_result)
            
            # Update performance stats
            self.performance_stats["total_workflows"] += 1
            self.performance_stats["memory_operations"] += 1
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "context": context_result["context"],
                "memory_summary": self._format_memory_summary(context_result["context"]),
                "recommendations": self._get_workflow_recommendations(agent_name, context_result["context"])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_id": None
            }
    
    def complete_agent_workflow(self, workflow_id: str, response_text: str,
                              explicit_delegations: Optional[List[Dict[str, str]]] = None,
                              completion_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Complete an agent workflow with automatic memory updates."""
        if workflow_id not in self.active_workflows:
            return {
                "success": False,
                "error": f"Workflow {workflow_id} not found or already completed"
            }
        
        workflow = self.active_workflows[workflow_id]
        
        try:
            # Calculate execution time
            execution_time = time.time() - workflow["start_time"]
            
            # Auto-update memory
            update_result = auto_update_agent_memory(
                agent_name=workflow["agent_name"],
                task_description=workflow["task_prompt"],
                response_text=response_text,
                execution_time=execution_time,
                explicit_delegations=explicit_delegations
            )
            
            # Update workflow status
            workflow["status"] = "completed"
            workflow["completion_time"] = time.time()
            workflow["update_result"] = update_result
            
            # Call completion callback if provided
            if completion_callback:
                completion_callback(workflow, update_result)
            
            # Update performance stats
            if update_result["success"]:
                self.performance_stats["successful_workflows"] += 1
                self.performance_stats["insights_generated"] += update_result.get("insights_extracted", 0)
            
            # Update average response time
            total_time = self.performance_stats["average_response_time"] * (self.performance_stats["total_workflows"] - 1)
            self.performance_stats["average_response_time"] = (total_time + execution_time) / self.performance_stats["total_workflows"]
            
            # Generate completion summary
            completion_summary = self._generate_completion_summary(workflow, update_result, execution_time)
            
            # Clean up workflow (keep last 50)
            if len(self.active_workflows) > 50:
                oldest_workflows = sorted(self.active_workflows.items(), 
                                        key=lambda x: x[1].get("completion_time", x[1]["start_time"]))[:10]
                for old_id, _ in oldest_workflows:
                    del self.active_workflows[old_id]
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "agent_name": workflow["agent_name"],
                "execution_time": execution_time,
                "update_result": update_result,
                "completion_summary": completion_summary,
                "performance_impact": self._assess_performance_impact(update_result)
            }
            
        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e),
                "workflow_id": workflow_id
            }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "agent_name": workflow["agent_name"],
            "status": workflow["status"],
            "task_prompt": workflow["task_prompt"][:100] + "..." if len(workflow["task_prompt"]) > 100 else workflow["task_prompt"],
            "elapsed_time": time.time() - workflow["start_time"],
            "context_loaded": "context" in workflow,
            "update_result": workflow.get("update_result")
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all currently active workflows."""
        active = []
        current_time = time.time()
        
        for workflow_id, workflow in self.active_workflows.items():
            if workflow["status"] == "active":
                active.append({
                    "workflow_id": workflow_id,
                    "agent_name": workflow["agent_name"],
                    "elapsed_time": current_time - workflow["start_time"],
                    "task_prompt": workflow["task_prompt"][:60] + "..." if len(workflow["task_prompt"]) > 60 else workflow["task_prompt"]
                })
        
        return active
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall workflow performance statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate success rate
        stats["success_rate"] = (stats["successful_workflows"] / max(stats["total_workflows"], 1)) * 100
        
        # Add memory efficiency metrics
        if stats["memory_operations"] > 0:
            stats["insights_per_operation"] = stats["insights_generated"] / stats["memory_operations"]
        
        # Add recent workflow data
        completed_workflows = [w for w in self.active_workflows.values() if w["status"] == "completed"]
        
        if completed_workflows:
            stats["recent_workflows"] = len(completed_workflows)
            stats["average_insights_per_workflow"] = sum(
                w.get("update_result", {}).get("insights_extracted", 0) for w in completed_workflows
            ) / len(completed_workflows)
        
        return stats
    
    def _format_memory_summary(self, context: Dict[str, Any]) -> str:
        """Format memory context into readable summary."""
        summary_parts = []
        
        # Recent performance
        recent_tasks = context.get("recent_tasks", [])
        if recent_tasks:
            success_count = sum(1 for t in recent_tasks if t.get("outcome") == "success")
            summary_parts.append(f"Recent Success Rate: {success_count}/{len(recent_tasks)} ({success_count/len(recent_tasks)*100:.0f}%)")
        
        # Key insights
        insights = context.get("optimization_insights", [])
        if insights:
            high_confidence = sum(1 for i in insights if i.get("confidence", 0) > 0.9)
            summary_parts.append(f"High-Confidence Insights: {high_confidence}/{len(insights)}")
        
        # Collaboration effectiveness
        collabs = context.get("collaboration_patterns", [])
        if collabs:
            best_collab = max(collabs, key=lambda x: x.get("success_rate", 0))
            summary_parts.append(f"Best Collaborator: {best_collab['agent_name']} ({best_collab['success_rate']*100:.0f}% success)")
        
        # Unread messages
        messages = context.get("unread_messages", [])
        if messages:
            urgent = sum(1 for m in messages if m.get("metadata", {}).get("priority") == "urgent")
            summary_parts.append(f"Unread Messages: {len(messages)} ({urgent} urgent)")
        
        return " | ".join(summary_parts) if summary_parts else "No significant memory patterns"
    
    def _get_workflow_recommendations(self, agent_name: str, context: Dict[str, Any]) -> List[str]:
        """Generate workflow recommendations based on agent memory."""
        recommendations = []
        
        # Based on collaboration patterns
        collabs = context.get("collaboration_patterns", [])
        for collab in collabs:
            if collab.get("success_rate", 0) > 0.9 and collab.get("collaboration_frequency", 0) > 3:
                recommendations.append(f"Consider delegating to {collab['agent_name']} (proven {collab['success_rate']*100:.0f}% success rate)")
        
        # Based on recent insights
        insights = context.get("optimization_insights", [])
        high_impact_areas = {}
        for insight in insights:
            if insight.get("impact") == "high":
                area = insight.get("area", "unknown")
                high_impact_areas[area] = high_impact_areas.get(area, 0) + 1
        
        if high_impact_areas:
            top_area = max(high_impact_areas.items(), key=lambda x: x[1])
            recommendations.append(f"Focus on {top_area[0].replace('_', ' ')} - historically high impact area")
        
        # Based on unread messages
        messages = context.get("unread_messages", [])
        urgent_messages = [m for m in messages if m.get("metadata", {}).get("priority") == "urgent"]
        if urgent_messages:
            recommendations.append(f"Address {len(urgent_messages)} urgent messages from other agents first")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _generate_completion_summary(self, workflow: Dict[str, Any], 
                                   update_result: Dict[str, Any], execution_time: float) -> str:
        """Generate a completion summary for the workflow."""
        agent = workflow["agent_name"]
        outcome = update_result.get("outcome", "unknown")
        insights = update_result.get("insights_extracted", 0)
        impact = update_result.get("impact_level", "medium")
        
        summary = f"{agent} completed task with {outcome} outcome in {execution_time:.1f}s"
        
        if insights > 0:
            summary += f", extracted {insights} insights"
        
        if impact == "high":
            summary += f" (HIGH IMPACT - {update_result.get('confidence', 0.8)*100:.0f}% confidence)"
        
        if update_result.get("delegations_identified", 0) > 0:
            summary += f", identified {update_result['delegations_identified']} collaboration opportunities"
        
        return summary
    
    def _assess_performance_impact(self, update_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the performance impact of a completed workflow."""
        impact_assessment = {
            "level": update_result.get("impact_level", "medium"),
            "confidence": update_result.get("confidence", 0.75),
            "insights_quality": "high" if update_result.get("insights_extracted", 0) >= 3 else "medium",
            "collaboration_potential": "high" if update_result.get("delegations_identified", 0) > 0 else "low",
            "learning_value": "high" if update_result.get("confidence", 0) > 0.9 else "medium"
        }
        
        # Calculate overall impact score
        scores = {
            "high": 3, "medium": 2, "low": 1
        }
        
        total_score = (scores[impact_assessment["level"]] + 
                      scores[impact_assessment["insights_quality"]] +
                      scores[impact_assessment["collaboration_potential"]] + 
                      scores[impact_assessment["learning_value"]])
        
        impact_assessment["overall_score"] = total_score
        impact_assessment["overall_rating"] = "excellent" if total_score >= 10 else "good" if total_score >= 8 else "standard"
        
        return impact_assessment


# Global workflow instance
memory_workflow = AgentMemoryWorkflow()


def start_memory_workflow(task_prompt: str, explicit_agent: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to start an agent memory workflow."""
    return memory_workflow.start_agent_workflow(task_prompt, explicit_agent)


def complete_memory_workflow(workflow_id: str, response_text: str, 
                            explicit_delegations: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Convenience function to complete an agent memory workflow."""
    return memory_workflow.complete_agent_workflow(workflow_id, response_text, explicit_delegations)


def get_workflow_performance() -> Dict[str, Any]:
    """Convenience function to get workflow performance stats."""
    return memory_workflow.get_performance_stats()


if __name__ == "__main__":
    # Test complete workflow integration
    print("🚀 Testing Complete Workflow Integration")
    print("=" * 55)
    
    workflow = AgentMemoryWorkflow()
    
    # Test workflow 1: Database optimization
    print("🔄 Starting Database Optimization Workflow...")
    start_result = workflow.start_agent_workflow(
        "Optimize the PostgreSQL query in our analytics dashboard that's taking 8 seconds to load"
    )
    
    if start_result["success"]:
        workflow_id = start_result["workflow_id"]
        print(f"✅ Workflow started: {workflow_id[:8]}...")
        print(f"   Agent: {start_result['agent_name']}")
        print(f"   Memory Summary: {start_result['memory_summary']}")
        print(f"   Recommendations: {len(start_result['recommendations'])}")
        
        for rec in start_result["recommendations"]:
            print(f"     • {rec}")
        
        # Simulate task completion
        response = """
        Analyzed the slow query using EXPLAIN and found it was doing a full table scan.
        Created a composite index on (user_id, session_date, metric_type) which eliminated the scan.
        Query performance improved dramatically - from 8.2 seconds to 0.3 seconds (96% improvement).
        Also optimized the connection pool settings and enabled query result caching.
        The dashboard now loads in under 1 second. Performance team should validate system-wide impact.
        """
        
        print("\n⚙️  Simulating task execution...")
        time.sleep(1)  # Simulate some execution time
        
        print("📤 Completing workflow with auto-memory update...")
        completion_result = workflow.complete_agent_workflow(workflow_id, response)
        
        if completion_result["success"]:
            print(f"✅ Workflow completed successfully")
            print(f"   Execution time: {completion_result['execution_time']:.1f}s")
            print(f"   Summary: {completion_result['completion_summary']}")
            print(f"   Impact: {completion_result['performance_impact']['overall_rating']}")
            print(f"   Insights extracted: {completion_result['update_result']['insights_extracted']}")
    
    # Test workflow 2: ML optimization
    print(f"\n{'='*55}")
    print("🔄 Starting ML Training Workflow...")
    
    ml_start = workflow.start_agent_workflow(
        "Improve the machine learning model accuracy for our prompt optimization system"
    )
    
    if ml_start["success"]:
        ml_id = ml_start["workflow_id"] 
        print(f"✅ ML Workflow started: {ml_id[:8]}...")
        print(f"   Agent: {ml_start['agent_name']}")
        
        ml_response = """
        Optimized the ML training pipeline with hyperparameter tuning and feature engineering.
        Model accuracy improved from 87% to 94% using advanced ensemble methods.
        Training time reduced by 40% through batch optimization and GPU acceleration.
        Implemented cross-validation and early stopping to prevent overfitting.
        Ready for production deployment - infrastructure team should handle scaling.
        """
        
        time.sleep(0.5)  # Simulate execution
        
        ml_completion = workflow.complete_agent_workflow(ml_id, ml_response)
        
        if ml_completion["success"]:
            print(f"✅ ML Workflow completed: {ml_completion['completion_summary']}")
    
    # Show overall performance
    print(f"\n{'='*55}")
    print("📊 Workflow Performance Statistics")
    print("=" * 35)
    
    stats = workflow.get_performance_stats()
    print(f"Total Workflows: {stats['total_workflows']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Avg Response Time: {stats['average_response_time']:.1f}s")
    print(f"Total Insights: {stats['insights_generated']}")
    print(f"Memory Operations: {stats['memory_operations']}")
    
    if stats.get("average_insights_per_workflow"):
        print(f"Avg Insights/Workflow: {stats['average_insights_per_workflow']:.1f}")
    
    print("\n✅ Complete workflow integration test successful!")