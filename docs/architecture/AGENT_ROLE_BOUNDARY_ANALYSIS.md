# Agent Role Boundary Analysis

## Executive Summary

This document analyzes role boundaries and overlaps between the 5 existing Claude Code agents and provides specific recommendations for optimization. The analysis identifies 4 primary overlap areas and proposes clear delegation patterns.

## Detailed Overlap Analysis

### 1. Database Performance Optimization

#### Current Overlap:
- **database-specialist**: Query optimization, indexing strategies, performance tuning
- **performance-engineer**: Database query profiling, slow query analysis, optimization recommendations

#### Conflict Scenarios:
- User asks: "This query is slow, how do I optimize it?"
- User asks: "My database performance is poor, what should I do?"
- User asks: "Should I add an index to this table?"

#### **RECOMMENDED RESOLUTION:**
**Primary Responsibility**: database-specialist
**Secondary Support**: performance-engineer

**Delegation Pattern:**
```
performance-engineer → database-specialist: Query optimization, indexing, schema changes
database-specialist → performance-engineer: System-wide performance impact assessment
```

**Clear Boundaries:**
- **database-specialist**: Query analysis, index design, schema optimization, database configuration
- **performance-engineer**: Performance measurement, system-wide bottleneck analysis, monitoring setup

### 2. ML Performance Optimization

#### Current Overlap:
- **ml-orchestrator**: Model optimization, training performance, hyperparameter tuning
- **performance-engineer**: ML component performance, resource utilization, bottleneck analysis

#### Conflict Scenarios:
- User asks: "My model training is too slow, how do I optimize it?"
- User asks: "How do I improve inference performance?"
- User asks: "My ML pipeline is using too much memory"

#### **RECOMMENDED RESOLUTION:**
**Primary Responsibility**: ml-orchestrator
**Secondary Support**: performance-engineer

**Delegation Pattern:**
```
performance-engineer → ml-orchestrator: ML algorithm optimization, model architecture changes
ml-orchestrator → performance-engineer: Infrastructure scaling, resource allocation, system monitoring
```

**Clear Boundaries:**
- **ml-orchestrator**: Model architecture, training algorithms, hyperparameters, ML-specific optimizations
- **performance-engineer**: Infrastructure performance, resource monitoring, system-level optimization

### 3. Security Infrastructure

#### Current Overlap:
- **security-architect**: Security policies, threat assessment, secure coding practices
- **infrastructure-specialist**: Security infrastructure setup, security tooling configuration

#### Conflict Scenarios:
- User asks: "How do I set up authentication for my API?"
- User asks: "What security monitoring should I implement?"
- User asks: "How do I configure rate limiting?"

#### **RECOMMENDED RESOLUTION:**
**Primary Responsibility**: security-architect
**Secondary Support**: infrastructure-specialist

**Delegation Pattern:**
```
infrastructure-specialist → security-architect: Security policy design, threat assessment
security-architect → infrastructure-specialist: Security tooling setup, infrastructure configuration
```

**Clear Boundaries:**
- **security-architect**: Security design, policies, threat modeling, secure coding practices
- **infrastructure-specialist**: Security tool deployment, configuration, infrastructure hardening

### 4. Monitoring and Observability

#### Current Overlap:
- **performance-engineer**: Performance monitoring, dashboards, alerting setup
- **infrastructure-specialist**: Monitoring infrastructure, tool configuration, data collection

#### Conflict Scenarios:
- User asks: "How do I set up monitoring for my application?"
- User asks: "What metrics should I track?"
- User asks: "How do I configure OpenTelemetry?"

#### **RECOMMENDED RESOLUTION:**
**Shared Responsibility with Clear Split**

**Delegation Pattern:**
```
performance-engineer: WHAT to monitor (metrics, thresholds, alerting logic)
infrastructure-specialist: HOW to monitor (tool setup, configuration, infrastructure)
```

**Clear Boundaries:**
- **performance-engineer**: Monitoring strategy, metrics selection, performance analysis, alerting logic
- **infrastructure-specialist**: Monitoring tool deployment, configuration, data collection infrastructure

## Recommended Agent Responsibility Matrix

| Domain | Primary Agent | Secondary Agent | Delegation Trigger |
|--------|---------------|-----------------|-------------------|
| **Database Query Optimization** | database-specialist | performance-engineer | "How do I optimize this query?" → database-specialist |
| **Database Performance Monitoring** | performance-engineer | database-specialist | "Set up database monitoring" → performance-engineer |
| **ML Model Optimization** | ml-orchestrator | performance-engineer | "Optimize my model" → ml-orchestrator |
| **ML Infrastructure Performance** | performance-engineer | ml-orchestrator | "Scale ML infrastructure" → performance-engineer |
| **Security Policy Design** | security-architect | infrastructure-specialist | "Design security for API" → security-architect |
| **Security Tool Setup** | infrastructure-specialist | security-architect | "Configure rate limiting" → infrastructure-specialist |
| **Monitoring Strategy** | performance-engineer | infrastructure-specialist | "What should I monitor?" → performance-engineer |
| **Monitoring Setup** | infrastructure-specialist | performance-engineer | "Configure OpenTelemetry" → infrastructure-specialist |

## Clear Delegation Patterns

### Pattern 1: Design vs Implementation
- **Design/Strategy**: security-architect (policies), performance-engineer (monitoring strategy), database-specialist (schema design)
- **Implementation/Setup**: infrastructure-specialist (tool deployment), performance-engineer (monitoring setup)

### Pattern 2: Domain Expertise
- **Database Domain**: database-specialist leads all database-specific optimizations
- **ML Domain**: ml-orchestrator leads all ML-specific optimizations  
- **Security Domain**: security-architect leads all security design decisions

### Pattern 3: System vs Component
- **System-Wide**: performance-engineer (cross-system optimization, monitoring)
- **Component-Specific**: Domain experts (database-specialist for DB, ml-orchestrator for ML)

## Updated Agent Descriptions

### Recommended Description Updates:

#### database-specialist
```markdown
PRIMARY: Database design, schema optimization, query performance, migration strategies
DELEGATES TO: performance-engineer for system-wide performance impact
RECEIVES FROM: performance-engineer for database bottleneck identification
```

#### performance-engineer  
```markdown
PRIMARY: System-wide performance analysis, monitoring strategy, bottleneck identification
DELEGATES TO: database-specialist for query optimization, ml-orchestrator for model optimization
RECEIVES FROM: All agents for performance measurement and monitoring setup
```

#### ml-orchestrator
```markdown  
PRIMARY: ML pipeline design, model optimization, training performance, MLOps
DELEGATES TO: performance-engineer for infrastructure scaling and resource allocation
RECEIVES FROM: performance-engineer for ML system bottleneck analysis
```

#### security-architect
```markdown
PRIMARY: Security design, policies, threat modeling, vulnerability assessment
DELEGATES TO: infrastructure-specialist for security tool deployment
RECEIVES FROM: infrastructure-specialist for security infrastructure requirements
```

#### infrastructure-specialist
```markdown
PRIMARY: Development environments, tool deployment, infrastructure setup
DELEGATES TO: security-architect for security design, performance-engineer for monitoring strategy
RECEIVES FROM: All agents for infrastructure and tooling requirements
```

## Implementation Recommendations

1. **Update Agent Descriptions**: Incorporate delegation patterns into agent markdown files
2. **Add Delegation Examples**: Include specific scenarios showing when to delegate
3. **Create Decision Tree**: Simple flowchart for users to understand which agent to use
4. **Test Delegation**: Validate patterns with real project scenarios

---

*Generated as part of Claude Code Agent Enhancement Project - Phase 1*