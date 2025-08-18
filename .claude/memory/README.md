# Agent Memory System

This directory contains JSON-based persistent memory for Claude Code specialized agents.

## Structure

```
.claude/memory/
├── README.md                           # This file
├── schemas/                           # JSON schemas for validation
│   ├── agent_memory_schema.json      # Individual agent memory schema
│   └── shared_context_schema.json    # Cross-agent context schema
├── agents/                           # Individual agent memories
│   ├── database-specialist.json
│   ├── ml-orchestrator.json
│   ├── performance-engineer.json
│   ├── security-architect.json
│   ├── infrastructure-specialist.json
│   ├── data-pipeline-specialist.json
│   ├── api-design-specialist.json
│   ├── monitoring-observability-specialist.json
│   ├── testing-strategy-specialist.json
│   ├── configuration-management-specialist.json
│   └── documentation-specialist.json
├── shared_context.json              # Cross-agent shared context
└── memory_manager.py               # Memory management utilities
```

## Usage

Each agent automatically loads its memory context before task execution and updates it afterward. The shared context enables cross-agent communication and context continuity.

## Memory Lifecycle

1. **Pre-task**: Agent loads relevant memories and shared context
2. **Task execution**: Agent processes task with full context
3. **Post-task**: Agent updates memories and shared context
4. **Cleanup**: Periodic pruning of outdated memories

## Schema Validation

All memory operations are validated against JSON schemas to ensure consistency and prevent corruption.