#!/usr/bin/env python3
"""
Memory Loading and Update Hooks

Provides automatic memory loading before agent task execution and memory updates
after task completion for seamless agent memory integration.
"""

import json
import os
import re
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any
from memory_manager import AgentMemoryManager


class AgentMemoryHooks:
    """Hooks for automatic agent memory loading and updates."""
    
    def __init__(self, memory_dir: str = ".claude/memory"):
        """Initialize memory hooks with directory path."""
        self.manager = AgentMemoryManager(memory_dir)
        self.memory_dir = Path(memory_dir)
        self.current_task_id = None
        self.current_agent = None
        self.task_start_time = None
    
    def detect_agent_from_context(self, task_prompt: str) -> Optional[str]:
        """Detect which agent is being used from task context."""
        # Agent detection patterns based on task content and delegation
        agent_patterns = {
            "database-specialist": [
                r"database.*optimization", r"query.*slow", r"slow.*query", r"postgresql", 
                r"migration", r"schema", r"index", r"connection.*pool", r"sql", 
                r"query.*optimization", r"database.*performance", r"takes.*seconds.*run"
            ],
            "ml-orchestrator": [
                r"machine.*learning", r"model.*train", r"ml.*pipeline", 
                r"feature.*engineering", r"hyperparameter", r"algorithm"
            ],
            "performance-engineer": [
                r"performance.*bottleneck", r"system.*slow", r"optimization", 
                r"caching", r"response.*time", r"throughput"
            ],
            "security-architect": [
                r"security.*review", r"authentication", r"authorization", 
                r"vulnerability", r"threat.*model", r"encryption"
            ],
            "infrastructure-specialist": [
                r"docker", r"container", r"deployment", r"testcontainer", 
                r"ci/cd", r"infrastructure"
            ],
            "data-pipeline-specialist": [
                r"data.*pipeline", r"etl", r"analytics.*data", 
                r"data.*processing", r"data.*quality"
            ],
            "api-design-specialist": [
                r"api.*endpoint", r"fastapi", r"openapi", r"rest.*api", 
                r"websocket", r"api.*design"
            ],
            "monitoring-observability-specialist": [
                r"monitoring", r"observability", r"opentelemetry", 
                r"slo", r"metrics", r"distributed.*tracing"
            ],
            "testing-strategy-specialist": [
                r"testing.*strategy", r"test.*coverage", r"real.*behavior.*test", 
                r"integration.*test", r"test.*quality"
            ],
            "configuration-management-specialist": [
                r"configuration", r"environment.*config", r"config.*management", 
                r"environment.*variable", r"secret.*management"
            ],
            "documentation-specialist": [
                r"documentation", r"api.*docs", r"technical.*writing", 
                r"architecture.*decision", r"knowledge.*management"
            ]
        }
        
        prompt_lower = task_prompt.lower()
        
        # Check for explicit agent mentions first
        for agent_name in agent_patterns:
            if agent_name.replace("-", " ") in prompt_lower or agent_name.replace("-", "_") in prompt_lower:
                return agent_name
        
        # Check for pattern matches
        for agent_name, patterns in agent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    return agent_name
        
        return None
    
    def pre_task_memory_load(self, task_prompt: str, explicit_agent: Optional[str] = None) -> Dict[str, Any]:
        """Load agent memory before task execution."""
        # Detect or use explicit agent
        agent_name = explicit_agent or self.detect_agent_from_context(task_prompt)
        
        if not agent_name:
            # Return empty context if no agent detected
            return {
                "agent_detected": False,
                "memory_loaded": False,
                "context": {}
            }
        
        self.current_agent = agent_name
        self.current_task_id = str(uuid.uuid4())
        self.task_start_time = datetime.now(UTC)
        
        # Load agent memory and shared context
        agent_memory = self.manager.load_agent_memory(agent_name)
        shared_context = self.manager.load_shared_context()
        
        # Get unread messages for this agent
        unread_messages = self.manager.get_unread_messages(agent_name)
        
        # Prepare context summary for agent
        context_summary = {
            "agent_name": agent_name,
            "task_id": self.current_task_id,
            "recent_tasks": agent_memory["task_history"][:5],
            "optimization_insights": agent_memory["optimization_insights"][-10:],  # Last 10
            "collaboration_patterns": agent_memory["collaboration_patterns"]["frequent_collaborators"],
            "unread_messages": unread_messages,
            "project_context": shared_context["project_context"],
            "global_insights": shared_context["global_insights"][-5:],  # Last 5
            "performance_baselines": shared_context.get("performance_baselines", {}),
            "memory_loaded_at": datetime.now(UTC).isoformat() + "Z"
        }
        
        # Mark messages as acknowledged
        for message in unread_messages:
            self.manager.acknowledge_message(message["message_id"], agent_name)
        
        return {
            "agent_detected": True,
            "agent_name": agent_name,
            "memory_loaded": True,
            "task_id": self.current_task_id,
            "context": context_summary
        }
    
    def post_task_memory_update(self, task_outcome: str, insights: List[str] = None, 
                               delegations: List[Dict[str, str]] = None,
                               optimization_area: Optional[str] = None,
                               optimization_insight: Optional[str] = None,
                               optimization_impact: str = "medium",
                               optimization_confidence: float = 0.85) -> bool:
        """Update agent memory after task completion."""
        if not self.current_agent or not self.current_task_id:
            return False
        
        try:
            # Calculate task duration
            task_duration = None
            if self.task_start_time:
                duration = datetime.now(UTC) - self.task_start_time
                task_duration = duration.total_seconds()
            
            # Record task in history
            task_info = {
                "task_id": self.current_task_id,
                "task_description": f"Agent task completed with {task_outcome} outcome",
                "outcome": task_outcome,
                "key_insights": insights or [],
                "delegations": delegations or [],
                "duration_seconds": task_duration
            }
            
            self.manager.add_task_to_history(self.current_agent, task_info)
            
            # Add optimization insight if provided
            if optimization_area and optimization_insight:
                insight_info = {
                    "area": optimization_area,
                    "insight": optimization_insight,
                    "impact": optimization_impact,
                    "confidence": optimization_confidence
                }
                self.manager.add_optimization_insight(self.current_agent, insight_info)
            
            # Update collaboration patterns for delegations
            for delegation in (delegations or []):
                success = delegation.get("outcome", "").lower() == "success"
                task_type = delegation.get("reason", "unknown")
                collaborator = delegation.get("to_agent")
                
                if collaborator:
                    self.manager.update_collaboration_pattern(
                        self.current_agent, collaborator, success, task_type
                    )
            
            # Reset current task tracking
            self.current_task_id = None
            self.current_agent = None
            self.task_start_time = None
            
            return True
            
        except Exception as e:
            print(f"Error updating memory: {e}")
            return False
    
    def send_cross_agent_insight(self, message_type: str, content: str, 
                                target_agents: List[str] = None,
                                priority: str = "medium",
                                metadata: Dict[str, Any] = None,
                                from_agent: Optional[str] = None) -> Optional[str]:
        """Send insight or context to other agents.
        
        Args:
            message_type: Type of message to send
            content: Message content
            target_agents: Optional list of target agents
            priority: Message priority (low/medium/high)
            metadata: Additional metadata
            from_agent: Optional sender agent (uses current_agent if not provided)
        
        Returns:
            Message ID if sent successfully, None otherwise
        """
        # Use provided from_agent or fall back to current_agent
        sender = from_agent or self.current_agent
        if not sender:
            return None
        
        full_metadata = metadata or {}
        full_metadata["priority"] = priority
        
        # Only add task_id if it exists
        if self.current_task_id:
            full_metadata["task_id"] = self.current_task_id
        
        return self.manager.add_inter_agent_message(
            from_agent=sender,
            message_type=message_type,
            content=content,
            target_agents=target_agents,
            metadata=full_metadata
        )
    
    def get_collaboration_recommendations(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collaboration recommendations based on memory patterns."""
        target_agent = agent_name or self.current_agent
        if not target_agent:
            return {}
        
        memory = self.manager.load_agent_memory(target_agent)
        collaborators = memory["collaboration_patterns"]["frequent_collaborators"]
        
        recommendations = {
            "most_effective_collaborators": [],
            "delegation_suggestions": [],
            "collaboration_insights": []
        }
        
        # Sort collaborators by success rate
        sorted_collaborators = sorted(
            collaborators, 
            key=lambda x: x.get("success_rate", 0.0), 
            reverse=True
        )
        
        for collab in sorted_collaborators[:3]:  # Top 3
            recommendations["most_effective_collaborators"].append({
                "agent": collab["agent_name"],
                "success_rate": collab["success_rate"],
                "frequency": collab["collaboration_frequency"],
                "common_tasks": collab.get("common_tasks", [])
            })
        
        # Generate delegation suggestions
        delegation_effectiveness = memory["collaboration_patterns"]["delegation_effectiveness"]
        for agent, stats in delegation_effectiveness.items():
            if stats.get("success_rate", 0) > 0.8:  # High success rate
                recommendations["delegation_suggestions"].append({
                    "delegate_to": agent,
                    "success_rate": stats["success_rate"],
                    "avg_resolution_time": stats.get("average_resolution_time", 0),
                    "recommended_for": stats.get("common_issues", [])
                })
        
        return recommendations


# Global hooks instance for easy access
memory_hooks = AgentMemoryHooks()


def load_agent_context(task_prompt: str, explicit_agent: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load agent context before task execution."""
    return memory_hooks.pre_task_memory_load(task_prompt, explicit_agent)


def update_agent_memory(task_outcome: str, insights: List[str] = None, 
                       delegations: List[Dict[str, str]] = None,
                       optimization_area: Optional[str] = None,
                       optimization_insight: Optional[str] = None,
                       optimization_impact: str = "medium",
                       optimization_confidence: float = 0.85) -> bool:
    """Convenience function to update agent memory after task completion."""
    return memory_hooks.post_task_memory_update(
        task_outcome, insights, delegations, optimization_area,
        optimization_insight, optimization_impact, optimization_confidence
    )


def send_agent_insight(message_type: str, content: str, 
                      target_agents: List[str] = None,
                      priority: str = "medium",
                      metadata: Dict[str, Any] = None) -> Optional[str]:
    """Convenience function to send cross-agent insights."""
    return memory_hooks.send_cross_agent_insight(
        message_type, content, target_agents, priority, metadata
    )


def get_agent_collaboration_tips(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get collaboration recommendations."""
    return memory_hooks.get_collaboration_recommendations(agent_name)


if __name__ == "__main__":
    # Test the memory hooks system
    hooks = AgentMemoryHooks()
    
    # Test agent detection
    test_prompts = [
        "Optimize this slow PostgreSQL query that's affecting performance",
        "Set up machine learning model training for better accuracy",
        "Review API security for authentication vulnerabilities", 
        "Configure Docker containers for testing environment"
    ]
    
    print("Testing agent detection:")
    for prompt in test_prompts:
        agent = hooks.detect_agent_from_context(prompt)
        print(f"  '{prompt[:50]}...' → {agent}")
    
    # Test memory loading
    print("\nTesting memory loading:")
    context = hooks.pre_task_memory_load("Database query optimization needed")
    print(f"  Agent detected: {context.get('agent_detected')}")
    print(f"  Agent name: {context.get('agent_name')}")
    print(f"  Memory loaded: {context.get('memory_loaded')}")
    
    # Test memory update
    print("\nTesting memory update:")
    success = hooks.post_task_memory_update(
        "success", 
        ["Query optimized with new index", "Response time improved by 60%"],
        [{"to_agent": "performance-engineer", "reason": "performance validation", "outcome": "success"}],
        "query_optimization",
        "Added composite index on frequently queried columns",
        "high",
        0.95
    )
    print(f"  Memory update successful: {success}")
    
    print("\n✅ Memory hooks system test completed!")