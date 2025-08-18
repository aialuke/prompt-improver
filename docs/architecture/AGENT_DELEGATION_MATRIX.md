# Agent Delegation Matrix & Clear Patterns

## Executive Summary

This document provides the definitive delegation patterns and responsibility matrix for the optimized Claude Code agent system. All role boundary overlaps have been resolved and clear delegation patterns established.

## Optimized Agent Responsibility Matrix

### Primary Responsibilities

| Agent | Primary Domain | Key Responsibilities | Delegates To | Receives From |
|-------|---------------|---------------------|--------------|---------------|
| **database-specialist** | Database Operations | Query optimization, indexing, schema design, migrations, database-specific performance | performance-engineer (system impact) | performance-engineer (bottleneck identification) |
| **ml-orchestrator** | ML Algorithms & Models | Model architecture, training optimization, hyperparameters, ML-specific algorithms | performance-engineer (infrastructure scaling) | performance-engineer (ML bottleneck identification) |
| **performance-engineer** | System-Wide Performance | Performance monitoring, bottleneck identification, system optimization, monitoring strategy | database-specialist (DB optimization), ml-orchestrator (ML optimization) | All agents (monitoring setup requests) |
| **security-architect** | Security Design & Policy | Threat modeling, security policies, vulnerability assessment, secure coding practices | infrastructure-specialist (security tool deployment) | infrastructure-specialist (security infrastructure requirements) |
| **infrastructure-specialist** | Infrastructure & Deployment | Development environments, CI/CD, testing infrastructure, tool deployment, security tool setup | security-architect (security design) | security-architect (security requirements), performance-engineer (monitoring strategy) |

## Detailed Delegation Patterns

### Pattern 1: Performance Optimization Hierarchy

```
User Request: "My application is slow"
├── performance-engineer (PRIMARY)
│   ├── Identifies bottleneck type
│   ├── System-wide performance analysis
│   └── Delegates to specialists:
│       ├── database-specialist (if database bottleneck)
│       ├── ml-orchestrator (if ML bottleneck)
│       └── infrastructure-specialist (if infrastructure bottleneck)
└── Maintains monitoring and validation oversight
```

**Decision Rules:**
- **General performance issue** → performance-engineer
- **"This query is slow"** → database-specialist  
- **"Model training is slow"** → ml-orchestrator
- **"Need performance monitoring"** → performance-engineer

### Pattern 2: Security Implementation Hierarchy

```
User Request: "Implement security for my API"
├── security-architect (PRIMARY)
│   ├── Defines security requirements
│   ├── Creates threat model
│   ├── Specifies authentication/authorization design
│   └── Delegates implementation:
│       └── infrastructure-specialist (tool deployment, configuration)
└── Validates security implementation meets requirements
```

**Decision Rules:**
- **Security design/policy** → security-architect
- **Security tool setup** → infrastructure-specialist
- **Vulnerability assessment** → security-architect
- **Rate limiting configuration** → infrastructure-specialist (with security-architect requirements)

### Pattern 3: Database vs System Performance

```
Database Performance Issue:
├── If query-specific:
│   └── database-specialist (PRIMARY)
│       ├── Query optimization
│       ├── Index design
│       └── Schema changes
└── If system-wide database impact:
    └── performance-engineer (PRIMARY)
        ├── System performance analysis
        ├── Resource allocation
        └── Delegates specific optimization → database-specialist
```

**Decision Rules:**
- **"Optimize this query"** → database-specialist
- **"Database is bottleneck in system"** → performance-engineer → database-specialist
- **"Set up database monitoring"** → performance-engineer
- **"Design database schema"** → database-specialist

### Pattern 4: ML Performance Optimization

```
ML Performance Issue:
├── If algorithm/model-specific:
│   └── ml-orchestrator (PRIMARY)
│       ├── Hyperparameter tuning
│       ├── Model architecture optimization
│       └── Algorithm selection
└── If infrastructure/resource-specific:
    └── performance-engineer (PRIMARY)
        ├── Resource allocation
        ├── Infrastructure scaling
        └── Delegates ML optimization → ml-orchestrator
```

**Decision Rules:**
- **"Model accuracy is poor"** → ml-orchestrator
- **"Training uses too much memory"** → performance-engineer → ml-orchestrator
- **"Set up ML monitoring"** → performance-engineer
- **"Optimize model architecture"** → ml-orchestrator

## Clear Delegation Trigger Phrases

### Database Domain
| User Input | Primary Agent | Delegation Pattern |
|------------|---------------|-------------------|
| "This query is slow" | database-specialist | Direct |
| "Database performance is poor" | performance-engineer | → database-specialist |
| "Need database monitoring" | performance-engineer | Collaboration |
| "Design database schema" | database-specialist | Direct |
| "Migration strategy" | database-specialist | Direct |

### ML Domain  
| User Input | Primary Agent | Delegation Pattern |
|------------|---------------|-------------------|
| "Model training is slow" | ml-orchestrator | Direct |
| "ML system uses too much CPU" | performance-engineer | → ml-orchestrator |
| "Need ML monitoring" | performance-engineer | Collaboration |
| "Optimize model architecture" | ml-orchestrator | Direct |
| "Scale ML infrastructure" | performance-engineer | Collaboration with ml-orchestrator |

### Security Domain
| User Input | Primary Agent | Delegation Pattern |
|------------|---------------|-------------------|
| "Review this auth code" | security-architect | Direct |
| "Set up rate limiting" | infrastructure-specialist | Receive requirements from security-architect |
| "Security threat assessment" | security-architect | Direct |
| "Configure security tools" | infrastructure-specialist | Receive design from security-architect |
| "Security monitoring setup" | infrastructure-specialist | Collaboration with security-architect |

### Performance Domain
| User Input | Primary Agent | Delegation Pattern |
|------------|---------------|-------------------|
| "Application is slow" | performance-engineer | Direct, then delegate specifics |
| "Need monitoring setup" | performance-engineer | Direct |
| "System bottleneck analysis" | performance-engineer | Direct |
| "Performance optimization" | performance-engineer | Direct, delegate domain-specific |

## Collaboration Patterns

### Design vs Implementation Pattern
- **Design/Strategy Agents**: security-architect (security design), performance-engineer (monitoring strategy), database-specialist (schema design)
- **Implementation/Setup Agents**: infrastructure-specialist (tool deployment), performance-engineer (monitoring setup)

### Domain Expertise Pattern  
- **Domain Leaders**: database-specialist (all DB), ml-orchestrator (all ML), security-architect (all security design)
- **System Integrator**: performance-engineer (cross-system optimization)
- **Infrastructure Provider**: infrastructure-specialist (all deployment and setup)

### Escalation Pattern
```
User → Primary Agent → Domain Specialist (if needed) → Back to Primary Agent (validation)
```

## Agent Interaction Examples

### Example 1: Slow Dashboard
```
User: "Our analytics dashboard is loading slowly"
└── performance-engineer (PRIMARY)
    ├── Analyze: Identifies database queries as bottleneck
    ├── Delegate: database-specialist for query optimization
    ├── Collaborate: Set up monitoring for ongoing tracking
    └── Validate: Confirm overall system performance improvement
```

### Example 2: API Security Implementation
```
User: "Secure our new API endpoints"
└── security-architect (PRIMARY)
    ├── Design: Authentication strategy, threat model
    ├── Delegate: infrastructure-specialist for tool deployment
    ├── Specify: Security requirements and policies
    └── Validate: Review final implementation
```

### Example 3: ML Training Performance
```
User: "ML training is consuming too many resources"
└── performance-engineer (PRIMARY)
    ├── Analyze: Resource utilization and system impact
    ├── Delegate: ml-orchestrator for algorithm optimization
    ├── Collaborate: Infrastructure scaling recommendations
    └── Monitor: Set up resource tracking and alerts
```

## Success Metrics

### Boundary Clarity
- ✅ Zero overlapping responsibilities
- ✅ Clear delegation triggers defined
- ✅ Escalation paths documented
- ✅ Collaboration patterns established

### Efficiency Improvements
- **Reduced decision time**: Clear agent selection criteria
- **Faster problem resolution**: Direct routing to domain experts
- **Better coverage**: No gaps in responsibility
- **Improved collaboration**: Structured interaction patterns

## Implementation Validation

All agent markdown files have been updated with:
- ✅ **Role Boundaries & Delegation** sections
- ✅ Clear **PRIMARY RESPONSIBILITY** definitions
- ✅ **DELEGATES TO** and **RECEIVES FROM** mappings
- ✅ **COLLABORATION** patterns specified
- ✅ Updated examples showing delegation scenarios

## Next Steps

Phase 2 Complete! Moving to Phase 3: MCP Integration Enhancement
- Activate PostgreSQL MCP server integration
- Set up monitoring MCP server
- Configure GitHub MCP server
- Validate enhanced agent capabilities

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 2 Complete*