#!/usr/bin/env python3
"""
Agent Memory Management System

Provides persistent memory storage and retrieval for Claude Code specialized agents
using JSON file-based local storage.
"""

import json
import os
import uuid
from datetime import datetime, UTC, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import jsonschema
from jsonschema import validate, ValidationError


class AgentMemoryManager:
    """Core memory management for Claude Code agents."""
    
    def __init__(self, memory_dir: str = ".claude/memory"):
        """Initialize memory manager with directory path."""
        self.memory_dir = Path(self._resolve_memory_directory(memory_dir))
        self.agents_dir = self.memory_dir / "agents"
        self.schemas_dir = self.memory_dir / "schemas"
        self.shared_context_file = self.memory_dir / "shared_context.json"
    
    def _resolve_memory_directory(self, memory_dir: str) -> Path:
        """Intelligently resolve memory directory path based on context."""
        memory_path = Path(memory_dir)
        
        # If absolute path and has schemas, use as-is
        if memory_path.is_absolute() and (memory_path / "schemas").exists():
            return memory_path
        
        # Check if we're already in a .claude/memory directory
        current_dir = Path.cwd()
        if current_dir.name == "memory" and current_dir.parent.name == ".claude":
            # We're already in the memory directory, so use current directory
            return current_dir
        
        # Check if current directory contains .claude/memory
        candidate_path = current_dir / memory_dir
        if candidate_path.exists() and (candidate_path / "schemas").exists():
            return candidate_path
        
        # Try going up directories to find .claude/memory
        for parent in current_dir.parents:
            candidate_path = parent / memory_dir
            if candidate_path.exists() and (candidate_path / "schemas").exists():
                return candidate_path
        
        # If given absolute path doesn't have schemas, but we can find existing schemas,
        # use the path with schemas for schema loading but preserve the directory structure
        if memory_path.is_absolute():
            # Find existing schemas to use
            existing_schemas_dir = self._find_existing_schemas_directory()
            if existing_schemas_dir:
                # Use the given path for agents/shared_context but existing schemas
                return memory_path  # Will handle schema loading separately
        
        # Fall back to original relative path (may fail, but preserves original behavior)
        return memory_path
    
    def _find_existing_schemas_directory(self) -> Optional[Path]:
        """Find existing schemas directory for fallback use."""
        current_dir = Path.cwd()
        
        # Check current directory and parents for .claude/memory/schemas
        for directory in [current_dir] + list(current_dir.parents):
            schemas_path = directory / ".claude/memory/schemas"
            if schemas_path.exists():
                return schemas_path
        return None
    
    @cached_property
    def agent_schema(self) -> Dict[str, Any]:
        """Lazily load and cache agent memory schema."""
        return self._load_agent_schema()
    
    @cached_property
    def shared_context_schema(self) -> Dict[str, Any]:
        """Lazily load and cache shared context schema."""
        return self._load_shared_context_schema()
    
    def _load_agent_schema(self) -> Dict[str, Any]:
        """Load agent memory schema with fallback logic."""
        schemas_dir = self._get_schemas_directory()
        try:
            with open(schemas_dir / "agent_memory_schema.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(f"Agent memory schema not found: {e}. Searched in: {schemas_dir}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid agent memory schema JSON: {e}")
    
    def _load_shared_context_schema(self) -> Dict[str, Any]:
        """Load shared context schema with fallback logic."""
        schemas_dir = self._get_schemas_directory()
        try:
            with open(schemas_dir / "shared_context_schema.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise RuntimeError(f"Shared context schema not found: {e}. Searched in: {schemas_dir}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid shared context schema JSON: {e}")
    
    def _get_schemas_directory(self) -> Path:
        """Get schemas directory with fallback logic."""
        schemas_dir = self.schemas_dir
        
        # If schemas don't exist in the configured directory, try to find existing ones
        if not schemas_dir.exists():
            existing_schemas_dir = self._find_existing_schemas_directory()
            if existing_schemas_dir:
                schemas_dir = existing_schemas_dir
        
        return schemas_dir
    
    def _validate_agent_memory(self, memory_data: Dict[str, Any]) -> None:
        """Validate agent memory data against schema."""
        try:
            validate(instance=memory_data, schema=self.agent_schema)
        except ValidationError as e:
            raise ValueError(f"Agent memory validation failed: {e.message}")
    
    def _validate_shared_context(self, context_data: Dict[str, Any]) -> None:
        """Validate shared context data against schema."""
        try:
            validate(instance=context_data, schema=self.shared_context_schema)
        except ValidationError as e:
            raise ValueError(f"Shared context validation failed: {e.message}")
    
    def _get_agent_file(self, agent_name: str) -> Path:
        """Get the file path for an agent's memory."""
        return self.agents_dir / f"{agent_name}.json"
    
    def load_agent_memory(self, agent_name: str) -> Dict[str, Any]:
        """Load memory for a specific agent."""
        agent_file = self._get_agent_file(agent_name)
        
        if not agent_file.exists():
            # Create default memory structure
            default_memory = {
                "agent_name": agent_name,
                "last_updated": datetime.now(UTC).isoformat() + "Z",
                "task_history": [],
                "domain_knowledge": {
                    "common_patterns": [],
                    "project_specific_context": {
                        "architecture_patterns": [],
                        "performance_baselines": {},
                        "configuration_insights": []
                    }
                },
                "optimization_insights": [],
                "collaboration_patterns": {
                    "frequent_collaborators": [],
                    "delegation_effectiveness": {}
                }
            }
            self.save_agent_memory(agent_name, default_memory)
            return default_memory
        
        try:
            with open(agent_file, 'r') as f:
                memory_data = json.load(f)
                self._validate_agent_memory(memory_data)
                return memory_data
        except (json.JSONDecodeError, ValidationError) as e:
            raise RuntimeError(f"Failed to load agent memory for {agent_name}: {e}")
    
    def save_agent_memory(self, agent_name: str, memory_data: Dict[str, Any]) -> None:
        """Save memory for a specific agent."""
        # Update timestamp
        memory_data["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        
        # Enforce consistent memory limits before saving
        if "task_history" in memory_data and isinstance(memory_data["task_history"], list):
            # Keep only last 50 tasks (consistent with add_task_to_history)
            memory_data["task_history"] = memory_data["task_history"][:50]
        
        if "optimization_insights" in memory_data and isinstance(memory_data["optimization_insights"], list):
            # Keep only last 20 insights (consistent with add_optimization_insight)
            memory_data["optimization_insights"] = memory_data["optimization_insights"][-20:]
        
        # Validate before saving
        self._validate_agent_memory(memory_data)
        
        agent_file = self._get_agent_file(agent_name)
        agent_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(agent_file, 'w') as f:
                json.dump(memory_data, f, indent=2, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save agent memory for {agent_name}: {e}")
    
    def add_task_to_history(self, agent_name: str, task_info: Dict[str, Any]) -> None:
        """Add a task to an agent's history."""
        memory = self.load_agent_memory(agent_name)
        
        # Ensure task has required fields
        task_entry = {
            "task_id": task_info.get("task_id", str(uuid.uuid4())),
            "timestamp": task_info.get("timestamp", datetime.now(UTC).isoformat() + "Z"),
            "task_description": task_info["task_description"],
            "outcome": task_info["outcome"],
            "key_insights": task_info.get("key_insights", []),
            "delegations": task_info.get("delegations", [])
        }
        
        # Add to history (keep last 50 tasks)
        memory["task_history"].insert(0, task_entry)
        memory["task_history"] = memory["task_history"][:50]
        
        self.save_agent_memory(agent_name, memory)
    
    def add_optimization_insight(self, agent_name: str, insight_info: Dict[str, Any]) -> None:
        """Add an optimization insight to an agent's memory."""
        memory = self.load_agent_memory(agent_name)
        
        insight_entry = {
            "area": insight_info["area"],
            "insight": insight_info["insight"],
            "impact": insight_info["impact"],
            "confidence": insight_info["confidence"],
            "discovered_at": insight_info.get("discovered_at", datetime.now(UTC).isoformat() + "Z")
        }
        
        memory["optimization_insights"].append(insight_entry)
        
        # Keep only last 20 insights
        memory["optimization_insights"] = memory["optimization_insights"][-20:]
        
        self.save_agent_memory(agent_name, memory)
    
    def update_collaboration_pattern(self, agent_name: str, collaborator: str, 
                                   success: bool, task_type: str) -> None:
        """Update collaboration patterns between agents."""
        memory = self.load_agent_memory(agent_name)
        
        collaborators = memory["collaboration_patterns"]["frequent_collaborators"]
        
        # Find existing collaborator or create new entry
        collaborator_entry = None
        for collab in collaborators:
            if collab["agent_name"] == collaborator:
                collaborator_entry = collab
                break
        
        if collaborator_entry is None:
            collaborator_entry = {
                "agent_name": collaborator,
                "collaboration_frequency": 0,
                "success_rate": 0.0,
                "common_tasks": []
            }
            collaborators.append(collaborator_entry)
        
        # Update collaboration metrics
        collaborator_entry["collaboration_frequency"] += 1
        
        # Update success rate (simple running average)
        current_successes = collaborator_entry["success_rate"] * (collaborator_entry["collaboration_frequency"] - 1)
        if success:
            current_successes += 1
        collaborator_entry["success_rate"] = current_successes / collaborator_entry["collaboration_frequency"]
        
        # Track common task types
        if task_type not in collaborator_entry["common_tasks"]:
            collaborator_entry["common_tasks"].append(task_type)
        
        self.save_agent_memory(agent_name, memory)
    
    def load_shared_context(self) -> Dict[str, Any]:
        """Load shared context accessible by all agents."""
        if not self.shared_context_file.exists():
            # Create default shared context
            default_context = {
                "last_updated": datetime.now(UTC).isoformat() + "Z",
                "project_context": {
                    "project_name": "prompt-improver",
                    "current_phase": "development",
                    "active_objectives": [],
                    "system_health": {
                        "overall_status": "unknown",
                        "performance_metrics": {},
                        "last_health_check": datetime.now(UTC).isoformat() + "Z"
                    }
                },
                "inter_agent_messages": [],
                "global_insights": [],
                "architectural_decisions": [],
                "performance_baselines": {}
            }
            self.save_shared_context(default_context)
            return default_context
        
        try:
            with open(self.shared_context_file, 'r') as f:
                context_data = json.load(f)
                self._validate_shared_context(context_data)
                return context_data
        except (json.JSONDecodeError, ValidationError) as e:
            raise RuntimeError(f"Failed to load shared context: {e}")
    
    def save_shared_context(self, context_data: Dict[str, Any]) -> None:
        """Save shared context."""
        # Update timestamp
        context_data["last_updated"] = datetime.now(UTC).isoformat() + "Z"
        
        # Validate before saving
        self._validate_shared_context(context_data)
        
        try:
            with open(self.shared_context_file, 'w') as f:
                json.dump(context_data, f, indent=2, sort_keys=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save shared context: {e}")
    
    def add_inter_agent_message(self, from_agent: str, message_type: str, 
                               content: str, target_agents: Optional[List[str]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a message to inter-agent communication."""
        context = self.load_shared_context()
        
        message_id = str(uuid.uuid4())
        message = {
            "message_id": message_id,
            "from_agent": from_agent,
            "target_agents": target_agents or [],
            "message_type": message_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "acknowledged_by": []
        }
        
        # Add to messages (keep last 100)
        context["inter_agent_messages"].insert(0, message)
        context["inter_agent_messages"] = context["inter_agent_messages"][:100]
        
        self.save_shared_context(context)
        return message_id
    
    def acknowledge_message(self, message_id: str, agent_name: str) -> None:
        """Mark a message as acknowledged by an agent."""
        context = self.load_shared_context()
        
        for message in context["inter_agent_messages"]:
            if message["message_id"] == message_id:
                # Check if already acknowledged
                for ack in message["acknowledged_by"]:
                    if ack["agent"] == agent_name:
                        return  # Already acknowledged
                
                message["acknowledged_by"].append({
                    "agent": agent_name,
                    "timestamp": datetime.now(UTC).isoformat() + "Z"
                })
                break
        
        self.save_shared_context(context)
    
    def get_unread_messages(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get unread messages for a specific agent."""
        context = self.load_shared_context()
        unread_messages = []
        
        for message in context["inter_agent_messages"]:
            # Check if message is for this agent (broadcast or targeted)
            if not message["target_agents"] or agent_name in message["target_agents"]:
                # Check if not acknowledged
                acknowledged = False
                for ack in message["acknowledged_by"]:
                    if ack["agent"] == agent_name:
                        acknowledged = True
                        break
                
                if not acknowledged:
                    unread_messages.append(message)
        
        return unread_messages
    
    def agent_memory_exists(self, agent_name: str) -> bool:
        """Check if memory file exists for specified agent."""
        return self._get_agent_file(agent_name).exists()
    
    def delete_agent_memory(self, agent_name: str) -> bool:
        """Safely delete agent memory file with validation."""
        agent_file = self._get_agent_file(agent_name)
        
        if not agent_file.exists():
            return False
        
        try:
            # Verify file is readable before deletion
            with open(agent_file, 'r') as f:
                json.load(f)
            
            agent_file.unlink()
            return True
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Failed to delete agent memory for {agent_name}: {e}")
    
    def list_agents(self) -> List[str]:
        """Return list of all agents with existing memory files."""
        if not self.agents_dir.exists():
            return []
        
        agent_files = list(self.agents_dir.glob("*.json"))
        return [agent_file.stem for agent_file in agent_files]
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for an agent's memory."""
        if not self.agent_memory_exists(agent_name):
            return {"error": f"No memory found for agent: {agent_name}"}
        
        memory = self.load_agent_memory(agent_name)
        
        stats = {
            "agent_name": agent_name,
            "last_updated": memory.get("last_updated"),
            "total_tasks": len(memory.get("task_history", [])),
            "total_insights": len(memory.get("optimization_insights", [])),
            "collaboration_count": len(memory.get("collaboration_patterns", {}).get("frequent_collaborators", [])),
            "task_outcomes": {},
            "recent_activity_days": 0,
            "insight_confidence_avg": 0.0,
            "most_active_areas": [],
            "collaboration_success_rates": {}
        }
        
        # Analyze task outcomes
        for task in memory.get("task_history", []):
            outcome = task.get("outcome", "unknown")
            stats["task_outcomes"][outcome] = stats["task_outcomes"].get(outcome, 0) + 1
        
        # Calculate recent activity
        if memory.get("task_history"):
            try:
                last_task_time = datetime.fromisoformat(memory["task_history"][0]["timestamp"].replace('Z', '+00:00'))
                stats["recent_activity_days"] = (datetime.now(UTC).replace(tzinfo=last_task_time.tzinfo) - last_task_time).days
            except (ValueError, KeyError):
                pass
        
        # Analyze insights
        insights = memory.get("optimization_insights", [])
        if insights:
            total_confidence = sum(insight.get("confidence", 0) for insight in insights)
            stats["insight_confidence_avg"] = total_confidence / len(insights)
            
            # Find most active areas
            area_counts = {}
            for insight in insights:
                area = insight.get("area", "unknown")
                area_counts[area] = area_counts.get(area, 0) + 1
            stats["most_active_areas"] = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Analyze collaboration effectiveness
        for collab in memory.get("collaboration_patterns", {}).get("frequent_collaborators", []):
            agent = collab["agent_name"]
            stats["collaboration_success_rates"][agent] = {
                "success_rate": collab.get("success_rate", 0),
                "frequency": collab.get("collaboration_frequency", 0)
            }
        
        return stats
    
    def cleanup_expired_data(self, days_to_keep: int = 30) -> int:
        """Clean up old data from memory files."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat() + "Z"
        
        total_cleaned = 0
        
        # Clean up shared context messages
        context = self.load_shared_context()
        original_count = len(context["inter_agent_messages"])
        
        context["inter_agent_messages"] = [
            msg for msg in context["inter_agent_messages"]
            if msg["timestamp"] > cutoff_iso
        ]
        
        cleaned_count = original_count - len(context["inter_agent_messages"])
        if cleaned_count > 0:
            self.save_shared_context(context)
        total_cleaned += cleaned_count
        
        # Clean up agent task histories
        agent_files = list(self.agents_dir.glob("*.json"))
        for agent_file in agent_files:
            agent_name = agent_file.stem
            memory = self.load_agent_memory(agent_name)
            
            original_tasks = len(memory["task_history"])
            memory["task_history"] = [
                task for task in memory["task_history"]
                if task["timestamp"] > cutoff_iso
            ]
            
            cleaned_tasks = original_tasks - len(memory["task_history"])
            if cleaned_tasks > 0:
                self.save_agent_memory(agent_name, memory)
            total_cleaned += cleaned_tasks
        
        return total_cleaned


# Convenience functions for agent integration
def load_my_memory(agent_name: str) -> Dict[str, Any]:
    """Convenience function for agents to load their own memory."""
    manager = AgentMemoryManager()
    return manager.load_agent_memory(agent_name)


def save_my_memory(agent_name: str, memory_data: Dict[str, Any]) -> None:
    """Convenience function for agents to save their own memory."""
    manager = AgentMemoryManager()
    manager.save_agent_memory(agent_name, memory_data)


def load_shared_context() -> Dict[str, Any]:
    """Convenience function to load shared context."""
    manager = AgentMemoryManager()
    return manager.load_shared_context()


def send_message_to_agents(from_agent: str, message_type: str, content: str,
                          target_agents: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function to send inter-agent messages."""
    manager = AgentMemoryManager()
    return manager.add_inter_agent_message(
        from_agent, message_type, content, target_agents, metadata
    )


if __name__ == "__main__":
    # Basic test functionality
    manager = AgentMemoryManager()
    
    # Test agent memory
    test_memory = manager.load_agent_memory("database-specialist")
    print(f"Loaded memory for database-specialist: {len(test_memory['task_history'])} tasks")
    
    # Test shared context
    context = manager.load_shared_context()
    print(f"Loaded shared context: {len(context['inter_agent_messages'])} messages")
    
    print("Memory system basic test completed successfully!")