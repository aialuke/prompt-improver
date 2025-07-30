# PostgreSQL Integration Plan for APES
## Adaptive Prompt Enhancement System Database Strategy

### Executive Summary

This document outlines the complete PostgreSQL integration strategy for the APES project, providing three implementation approaches with detailed analysis, code examples, and step-by-step implementation guidance.

## Current APES Architecture Analysis

**Existing Components:**
- **MCP Server** (`src/prompt_improver/mcp_server/mcp_server.py`) - Model Context Protocol implementation
- **Rule Engine** (`src/prompt_improver/rule_engine/`) - Core rule processing
- **ML Optimizer** (`src/prompt_improver/rule_engine/ml_optimizer/`) - Machine learning optimization
- **CLI Interface** (`src/prompt_improver/cli.py`) - Command-line interface
- **MLflow Integration** - Current experiment tracking
- **Configuration Management** - YAML-based rule and ML configs

**Current Data Flow:**
```
User Input → MCP Server → Rule Engine → ML Optimizer → MLflow
                                     ↓
                                Rule Application
```

**Identified Limitations:**
1. **Static Rule Selection** - Rules applied based on configuration priority, not effectiveness data
2. **Limited Feedback Loop** - No systematic user feedback collection and analysis
3. **Isolated ML Training** - MLflow experiments don't influence real-time rule selection
4. **No Performance Analytics** - Missing rule effectiveness tracking and optimization insights

## Integration Strategy: Three Approaches

### Option A: MCP Server Only (Standardized Approach)

**Use Case:** External tool integration, development/testing, simple configurations

**Architecture:**
```
APES Components → MCP PostgreSQL Server → PostgreSQL Database
```

**Pros:**
- ✅ Standardized MCP protocol interface
- ✅ Security controls built-in
- ✅ Easy external tool integration (Claude Desktop, IDEs)
- ✅ Protocol-level validation and error handling

**Cons:**
- ❌ Protocol overhead for high-frequency operations
- ❌ Limited transaction control for complex ML workflows
- ❌ Potential performance bottlenecks for real-time rule selection

**Implementation:**
```python
# src/prompt_improver/database/mcp_client.py
from mcp import Client

class MCPPostgreSQLClient:
    def __init__(self, mcp_server_url):
        self.client = Client(mcp_server_url)
    
    async def record_rule_performance(self, rule_id: str, score: float, prompt_type: str):
        """Record rule effectiveness via MCP"""
        query = """
        INSERT INTO rule_performance (rule_id, improvement_score, prompt_type) 
        VALUES ($1, $2, $3)
        """
        return await self.client.call_tool("execute_query", {
            "query": query,
            "parameters": [rule_id, score, prompt_type]
        })
    
    async def get_best_rules(self, prompt_type: str, limit: int = 3):
        """Get top performing rules for prompt type"""
        query = """
        SELECT rule_id, avg_improvement 
        FROM rule_effectiveness_summary 
        WHERE prompt_types_count > 5 AND $1 = ANY(string_to_array($1, ','))
        ORDER BY avg_improvement DESC 
        LIMIT $2
        """
        return await self.client.call_tool("execute_query", {
            "query": query,
            "parameters": [prompt_type, limit]
        })
```

### Option B: Direct Database Access (Performance Approach)

**Use Case:** Production deployment, high-performance requirements, complex ML workflows

**Architecture:**
```
APES Components → asyncpg/SQLAlchemy → PostgreSQL Database
```

**Pros:**
- ✅ Maximum performance for frequent operations
- ✅ Full transaction control for ML workflows
- ✅ Advanced PostgreSQL features (JSONB, custom types, functions)
- ✅ Optimal for real-time rule selection

**Cons:**
- ❌ No standardized interface for external tools
- ❌ More complex database management
- ❌ Requires direct database security management

**Implementation:**
```python
# src/prompt_improver/database/postgres_client.py
import asyncpg
from typing import List, Dict, Optional
import json

class DirectPostgreSQLClient:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=2,
            max_size=20,
            command_timeout=30
        )
    
    async def record_rule_performance(
        self, 
        rule_id: str, 
        score: float, 
        prompt_type: str,
        confidence: float,
        execution_time_ms: int,
        prompt_characteristics: Dict,
        before_metrics: Dict,
        after_metrics: Dict
    ):
        """Record comprehensive rule performance data"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO rule_performance (
                    rule_id, rule_name, prompt_type, improvement_score, 
                    confidence_level, execution_time_ms, prompt_characteristics,
                    before_metrics, after_metrics
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, 
                rule_id, 
                self._get_rule_name(rule_id),
                prompt_type, 
                score, 
                confidence,
                execution_time_ms,
                json.dumps(prompt_characteristics),
                json.dumps(before_metrics),
                json.dumps(after_metrics)
            )
    
    async def get_optimal_rules(
        self, 
        prompt_characteristics: Dict, 
        limit: int = 3
    ) -> List[Dict]:
        """Get optimal rules based on prompt characteristics and historical performance"""
        async with self.pool.acquire() as conn:
            # Dynamic rule selection based on multiple factors
            rows = await conn.fetch("""
                SELECT 
                    rule_id,
                    rule_name,
                    avg_improvement,
                    confidence_score,
                    usage_count
                FROM rule_effectiveness_summary
                WHERE 
                    prompt_types_count > 3 
                    AND avg_confidence > 0.7
                    AND ($1::text IS NULL OR prompt_type = $1)
                ORDER BY 
                    avg_improvement DESC,
                    confidence_score DESC,
                    usage_count DESC
                LIMIT $2
            """, prompt_characteristics.get('type'), limit)
            
            return [dict(row) for row in rows]
    
    async def store_user_feedback(
        self,
        original_prompt: str,
        improved_prompt: str,
        user_rating: int,
        applied_rules: List[str],
        user_context: Dict,
        session_id: str
    ):
        """Store user feedback for continuous learning"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO user_feedback (
                    original_prompt, improved_prompt, user_rating,
                    applied_rules, user_context, session_id
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
                original_prompt,
                improved_prompt,
                user_rating,
                json.dumps(applied_rules),
                json.dumps(user_context),
                session_id
            )
    
    async def get_rule_analytics(self, days: int = 30) -> Dict:
        """Get comprehensive rule analytics for ML optimization"""
        async with self.pool.acquire() as conn:
            # Rule performance trends
            performance_data = await conn.fetch("""
                SELECT 
                    rule_id,
                    DATE_TRUNC('day', created_at) as date,
                    AVG(improvement_score) as avg_score,
                    COUNT(*) as usage_count,
                    AVG(confidence_level) as avg_confidence
                FROM rule_performance 
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY rule_id, DATE_TRUNC('day', created_at)
                ORDER BY date DESC
            """ % days)
            
            # User satisfaction correlation
            satisfaction_data = await conn.fetch("""
                SELECT 
                    jsonb_array_elements_text(applied_rules) as rule_id,
                    AVG(user_rating::float) as avg_rating,
                    COUNT(*) as feedback_count
                FROM user_feedback 
                WHERE created_at >= NOW() - INTERVAL '%s days'
                GROUP BY jsonb_array_elements_text(applied_rules)
                HAVING COUNT(*) >= 3
            """ % days)
            
            return {
                'performance_trends': [dict(row) for row in performance_data],
                'user_satisfaction': [dict(row) for row in satisfaction_data]
            }
```

### Option C: Hybrid Approach (Recommended)

**Use Case:** Production deployment with external integrations, best of both worlds

**Architecture:**
```
Internal Components → Direct DB Access → PostgreSQL Database
                                        ↑
External Tools → MCP Server ────────────┘
```

**Pros:**
- ✅ High performance for internal operations
- ✅ Standardized interface for external tools
- ✅ Full transaction control where needed
- ✅ Clean separation of concerns
- ✅ Optimal for both development and production

**Cons:**
- ❌ More complex initial setup
- ❌ Two codepaths to maintain

**Implementation:**
```python
# src/prompt_improver/database/hybrid_client.py
from .postgres_client import DirectPostgreSQLClient
from .mcp_client import MCPPostgreSQLClient
from typing import Optional, Union

class HybridDatabaseClient:
    """
    Hybrid client that uses direct access for internal operations
    and MCP for external integrations
    """
    
    def __init__(
        self, 
        connection_string: str,
        mcp_server_url: Optional[str] = None,
        prefer_direct: bool = True
    ):
        self.direct_client = DirectPostgreSQLClient(connection_string)
        self.mcp_client = MCPPostgreSQLClient(mcp_server_url) if mcp_server_url else None
        self.prefer_direct = prefer_direct
    
    async def initialize(self):
        """Initialize both clients"""
        await self.direct_client.initialize()
        if self.mcp_client:
            await self.mcp_client.initialize()
    
    def _should_use_direct(self, operation_type: str) -> bool:
        """Determine which client to use based on operation type"""
        high_performance_operations = {
            'record_rule_performance',
            'get_optimal_rules', 
            'batch_operations',
            'ml_analytics'
        }
        
        return (
            self.prefer_direct and 
            operation_type in high_performance_operations and
            self.direct_client.pool is not None
        )
    
    async def record_rule_performance(self, **kwargs):
        """Route to appropriate client based on performance needs"""
        if self._should_use_direct('record_rule_performance'):
            return await self.direct_client.record_rule_performance(**kwargs)
        elif self.mcp_client:
            return await self.mcp_client.record_rule_performance(**kwargs)
        else:
            raise RuntimeError("No available database client")
    
    async def get_optimal_rules(self, **kwargs):
        """Use direct client for real-time rule selection performance"""
        if self._should_use_direct('get_optimal_rules'):
            return await self.direct_client.get_optimal_rules(**kwargs)
        elif self.mcp_client:
            return await self.mcp_client.get_best_rules(**kwargs)
        else:
            raise RuntimeError("No available database client")
```

## Detailed Implementation Plan

### Phase 1: Database Infrastructure (Completed ✅)

**Already Implemented:**
- ✅ Docker PostgreSQL setup with persistent storage
- ✅ Comprehensive database schema with 8 optimized tables
- ✅ Management scripts and configuration
- ✅ Initialization and backup procedures

**Database Schema Highlights:**
- **rule_performance** - Rule effectiveness tracking with JSONB metadata
- **user_feedback** - User rating and satisfaction data
- **ml_model_performance** - ML model metrics linked to MLflow
- **discovered_patterns** - ML-discovered rule patterns
- **rule_combinations** - A/B testing for rule combinations
- **Optimized indexes** and **views** for analytics queries

### Phase 2: Integration Layer Implementation

#### 2.1 Database Client Setup

**For Hybrid Approach (Recommended):**

```python
# src/prompt_improver/database/__init__.py
from .hybrid_client import HybridDatabaseClient
from .postgres_client import DirectPostgreSQLClient
from .mcp_client import MCPPostgreSQLClient

__all__ = ['HybridDatabaseClient', 'DirectPostgreSQLClient', 'MCPPostgreSQLClient']

# src/prompt_improver/database/config.py
from pydantic import BaseSettings
from typing import Optional

class DatabaseConfig(BaseSettings):
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "apes_production"
    postgres_username: str = "apes_user"
    postgres_password: str = "apes_secure_password_2024"
    
    # MCP settings
    mcp_server_enabled: bool = True
    mcp_server_url: Optional[str] = None
    
    # Performance settings
    prefer_direct_access: bool = True
    connection_pool_min: int = 2
    connection_pool_max: int = 20
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_username}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
    
    class Config:
        env_file = ".env"
```

#### 2.2 Rule Engine Integration

**Update Rule Engine to Use Database:**

```python
# src/prompt_improver/rule_engine/enhanced_rule_engine.py
from typing import List, Dict, Optional
from .base import BasePromptRule
from ..database import HybridDatabaseClient
import time
import asyncio

class EnhancedRuleEngine:
    """
    Rule engine with PostgreSQL integration for dynamic rule selection
    and performance tracking
    """
    
    def __init__(self, db_client: HybridDatabaseClient):
        self.db_client = db_client
        self.rules = {}  # rule_id -> rule_instance
        self._load_rules()
    
    def _load_rules(self):
        """Load available rules from rule registry"""
        # Import and register all available rules
        from .rules.clarity import ClarityRule
        from .rules.specificity import SpecificityRule
        # Add other rules...
        
        self.rules = {
            'clarity_rule': ClarityRule(),
            'specificity_rule': SpecificityRule(),
            # Add other rule instances...
        }
    
    async def improve_prompt(
        self, 
        prompt: str, 
        user_context: Optional[Dict] = None,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Improve prompt using data-driven rule selection
        """
        start_time = time.time()
        
        # 1. Analyze prompt characteristics
        prompt_characteristics = self._analyze_prompt(prompt)
        
        # 2. Get optimal rules from database based on historical performance
        optimal_rules = await self.db_client.get_optimal_rules(
            prompt_characteristics=prompt_characteristics,
            limit=5
        )
        
        # 3. Apply rules in order of effectiveness
        improved_prompt = prompt
        applied_rules = []
        rule_performance_data = []
        
        for rule_data in optimal_rules:
            rule_id = rule_data['rule_id']
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                
                # Check if rule applies
                if rule.check(improved_prompt):
                    rule_start = time.time()
                    
                    # Apply rule
                    result = rule.apply(improved_prompt)
                    if result.success:
                        # Calculate improvement metrics
                        before_metrics = self._calculate_metrics(improved_prompt)
                        after_metrics = self._calculate_metrics(result.improved_prompt)
                        improvement_score = self._calculate_improvement_score(
                            before_metrics, after_metrics
                        )
                        
                        rule_execution_time = int((time.time() - rule_start) * 1000)
                        
                        # Store performance data for database
                        rule_performance_data.append({
                            'rule_id': rule_id,
                            'improvement_score': improvement_score,
                            'confidence': result.confidence,
                            'execution_time_ms': rule_execution_time,
                            'before_metrics': before_metrics,
                            'after_metrics': after_metrics
                        })
                        
                        improved_prompt = result.improved_prompt
                        applied_rules.append({
                            'rule_id': rule_id,
                            'improvement_score': improvement_score,
                            'confidence': result.confidence
                        })
        
        total_time = int((time.time() - start_time) * 1000)
        
        # 4. Store performance data asynchronously
        asyncio.create_task(self._store_performance_data(
            rule_performance_data,
            prompt_characteristics,
            session_id
        ))
        
        return {
            'original_prompt': prompt,
            'improved_prompt': improved_prompt,
            'applied_rules': applied_rules,
            'processing_time_ms': total_time,
            'session_id': session_id,
            'improvement_summary': self._generate_improvement_summary(applied_rules)
        }
    
    async def _store_performance_data(
        self, 
        rule_performance_data: List[Dict],
        prompt_characteristics: Dict,
        session_id: Optional[str]
    ):
        """Store rule performance data in database"""
        for data in rule_performance_data:
            try:
                await self.db_client.record_rule_performance(
                    rule_id=data['rule_id'],
                    score=data['improvement_score'],
                    prompt_type=prompt_characteristics.get('type', 'general'),
                    confidence=data['confidence'],
                    execution_time_ms=data['execution_time_ms'],
                    prompt_characteristics=prompt_characteristics,
                    before_metrics=data['before_metrics'],
                    after_metrics=data['after_metrics']
                )
            except Exception as e:
                # Log error but don't fail the main operation
                print(f"Failed to store performance data for {data['rule_id']}: {e}")
    
    def _analyze_prompt(self, prompt: str) -> Dict:
        """Analyze prompt characteristics for rule selection"""
        # Implementation for prompt analysis
        return {
            'type': self._classify_prompt_type(prompt),
            'length': len(prompt),
            'complexity': self._calculate_complexity(prompt),
            'domain': self._detect_domain(prompt),
            'clarity_score': self._assess_clarity(prompt),
            'specificity_score': self._assess_specificity(prompt)
        }
    
    def _calculate_improvement_score(self, before: Dict, after: Dict) -> float:
        """Calculate improvement score based on metrics"""
        # Weighted improvement calculation
        weights = {
            'clarity': 0.3,
            'specificity': 0.3,
            'completeness': 0.2,
            'structure': 0.2
        }
        
        total_improvement = 0
        for metric, weight in weights.items():
            before_score = before.get(metric, 0)
            after_score = after.get(metric, 0)
            improvement = (after_score - before_score) / max(before_score, 0.1)
            total_improvement += improvement * weight
        
        return max(0, min(1, total_improvement))  # Clamp to [0, 1]
```

#### 2.3 ML Optimizer Integration

**Enhance ML Optimizer with Database Analytics:**

```python
# src/prompt_improver/rule_engine/ml_optimizer/enhanced_optimizer.py
from ..database import HybridDatabaseClient
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import mlflow

class EnhancedMLOptimizer:
    """
    ML optimizer with PostgreSQL integration for advanced analytics
    and rule effectiveness prediction
    """
    
    def __init__(self, db_client: HybridDatabaseClient):
        self.db_client = db_client
    
    async def analyze_rule_effectiveness(self, days: int = 30) -> Dict:
        """
        Comprehensive rule effectiveness analysis using database data
        """
        # Get analytics data from database
        analytics_data = await self.db_client.get_rule_analytics(days=days)
        
        # Combine performance and satisfaction data
        performance_df = pd.DataFrame(analytics_data['performance_trends'])
        satisfaction_df = pd.DataFrame(analytics_data['user_satisfaction'])
        
        # Merge datasets
        combined_data = performance_df.merge(
            satisfaction_df, 
            on='rule_id', 
            how='left'
        ).fillna(0)
        
        # Calculate composite effectiveness score
        combined_data['effectiveness_score'] = (
            combined_data['avg_score'] * 0.4 +
            combined_data['avg_rating'] / 5 * 0.4 +  # Normalize rating to [0,1]
            np.log1p(combined_data['usage_count']) / 10 * 0.2  # Usage popularity
        )
        
        return {
            'rule_rankings': combined_data.sort_values(
                'effectiveness_score', ascending=False
            ).to_dict('records'),
            'trends': self._analyze_trends(combined_data),
            'recommendations': self._generate_recommendations(combined_data)
        }
    
    async def optimize_rule_parameters(self, rule_id: str) -> Dict:
        """
        Optimize rule parameters based on historical performance data using RuleOptimizer
        """
        # Get rule-specific performance data for RuleOptimizer
        performance_data = await self.get_rule_performance_data(rule_id)
        if not performance_data:
            return {'status': 'insufficient_data'}

        # Get historical data for advanced optimization
        historical_data = await self.get_historical_data(rule_id)

        # Use the actual RuleOptimizer.optimize_rule method
        optimization_result = await self.rule_optimizer.optimize_rule(
            rule_id=rule_id,
            performance_data=performance_data,
            historical_data=historical_data
        )

        # Track optimization experiment in MLflow
        with mlflow.start_run(run_name=f"rule_optimization_{rule_id}"):
            mlflow.log_params(optimization_result.get('optimized_parameters', {}))
            mlflow.log_metric("optimization_status", 1 if optimization_result['status'] == 'optimized' else 0)
            if 'multi_objective_optimization' in optimization_result:
                mlflow.log_metric("pareto_solutions", len(optimization_result['multi_objective_optimization'].get('pareto_frontier', [])))
            if 'gaussian_process_optimization' in optimization_result:
                mlflow.log_metric("gp_predicted_performance", optimization_result['gaussian_process_optimization'].get('predicted_performance', 0))

        return optimization_result

    async def get_rule_performance_data(self, rule_id: str) -> Dict:
        """Get performance data in the format expected by RuleOptimizer"""
        async with self.db_client.direct_client.pool.acquire() as conn:
            performance_data = await conn.fetch("""
                SELECT
                    COUNT(*) as total_applications,
                    AVG(improvement_score) as avg_improvement,
                    STDDEV(improvement_score) as improvement_stddev,
                    AVG(confidence_level) as avg_confidence
                FROM rule_performance
                WHERE rule_id = $1
                AND created_at >= NOW() - INTERVAL '90 days'
            """, rule_id)

        if not performance_data or not performance_data[0]['total_applications']:
            return {}

        row = performance_data[0]
        return {
            rule_id: {
                "total_applications": row['total_applications'],
                "avg_improvement": float(row['avg_improvement'] or 0),
                "consistency_score": 1.0 - float(row['improvement_stddev'] or 0.1),
                "confidence_level": float(row['avg_confidence'] or 0.5)
            }
        }

    async def get_historical_data(self, rule_id: str) -> List[Dict]:
        """Get historical data in the format expected by RuleOptimizer"""
        async with self.db_client.direct_client.pool.acquire() as conn:
            historical_data = await conn.fetch("""
                SELECT
                    improvement_score as score,
                    prompt_characteristics as context,
                    created_at as timestamp,
                    rule_parameters
                FROM rule_performance
                WHERE rule_id = $1
                AND created_at >= NOW() - INTERVAL '90 days'
                ORDER BY created_at DESC
                LIMIT 1000
            """, rule_id)

        return [
            {
                'score': float(row['score']),
                'context': row['context'] or {},
                'timestamp': row['timestamp'].isoformat(),
                'rule_parameters': row['rule_parameters'] or {}
            }
            for row in historical_data
        ]
    
    async def discover_rule_patterns(self) -> List[Dict]:
        """
        Discover new rule patterns from successful prompt improvements
        """
        # Get high-performing prompt improvements
        async with self.db_client.direct_client.pool.acquire() as conn:
            success_data = await conn.fetch("""
                SELECT 
                    uf.original_prompt,
                    uf.improved_prompt,
                    uf.user_rating,
                    uf.applied_rules,
                    rp.improvement_score,
                    rp.prompt_characteristics
                FROM user_feedback uf
                JOIN rule_performance rp ON uf.session_id = rp.prompt_id::text
                WHERE uf.user_rating >= 4 
                AND rp.improvement_score > 0.7
                AND uf.created_at >= NOW() - INTERVAL '30 days'
                ORDER BY uf.user_rating DESC, rp.improvement_score DESC
                LIMIT 500
            """)
        
        # Analyze patterns in successful improvements
        patterns = self._mine_patterns(success_data)
        
        # Store discovered patterns
        for pattern in patterns:
            await self._store_discovered_pattern(pattern)
        
        return patterns
    
    def _mine_patterns(self, success_data) -> List[Dict]:
        """Mine patterns from successful prompt improvements"""
        # Pattern mining implementation
        # This would include text analysis, similarity detection,
        # and rule pattern extraction
        patterns = []
        
        # Example pattern detection logic
        for row in success_data:
            original = row['original_prompt']
            improved = row['improved_prompt']
            
            # Detect common transformation patterns
            pattern = self._extract_transformation_pattern(original, improved)
            if pattern and self._validate_pattern(pattern):
                patterns.append({
                    'pattern_name': pattern['name'],
                    'pattern_rule': pattern['rule'],
                    'effectiveness_score': row['improvement_score'],
                    'user_satisfaction': row['user_rating'] / 5,
                    'discovery_method': 'text_analysis',
                    'sample_size': 1
                })
        
        # Consolidate similar patterns
        return self._consolidate_patterns(patterns)
```

#### 2.4 CLI and MCP Integration

**Update CLI to Use Database:**

```python
# src/prompt_improver/cli.py (enhanced)
from .database import HybridDatabaseClient, DatabaseConfig
from .rule_engine.enhanced_rule_engine import EnhancedRuleEngine
from .rule_engine.ml_optimizer.enhanced_optimizer import EnhancedMLOptimizer
import uuid
from typing import Optional
import typer
from rich.console import Console

console = Console()
app = typer.Typer()

# Initialize database client
db_config = DatabaseConfig()
db_client = HybridDatabaseClient(
    connection_string=db_config.postgres_url,
    prefer_direct=db_config.prefer_direct_access
)
rule_engine = EnhancedRuleEngine(db_client)
ml_optimizer = EnhancedMLOptimizer(db_client)

@app.command()
async def improve_prompt(
    prompt: str,
    session_id: Optional[str] = None
):
    """
    Improve a prompt using data-driven rule selection
    """
    try:
        # Initialize database connection
        await db_client.initialize()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Improve prompt using enhanced rule engine
        result = await rule_engine.improve_prompt(
            prompt=prompt,
            session_id=session_id
        )
        
        console.print(f"\n[green]Improved Prompt:[/green]")
        console.print(result['improved_prompt'])
        console.print(f"\n[blue]Applied Rules:[/blue] {', '.join(result['applied_rules'])}")
        console.print(f"[blue]Processing Time:[/blue] {result['processing_time_ms']}ms")
        console.print(f"[blue]Session ID:[/blue] {result['session_id']}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

@app.command()
async def analytics(
    days: int = 30
):
    """
    Get rule effectiveness analytics
    """
    try:
        await db_client.initialize()
        analytics = await ml_optimizer.analyze_rule_effectiveness(days=days)
        
        console.print(f"\n[green]Rule Effectiveness Analytics (Last {days} days):[/green]")
        for rule_id, metrics in analytics.items():
            console.print(f"  {rule_id}: {metrics['avg_improvement']:.2f} avg improvement")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

@app.command()
async def optimize_rules():
    """
    Trigger ML optimization of rule parameters
    """
    try:
        await db_client.initialize()
        console.print("[yellow]Starting ML optimization...[/yellow]")
        
        # Discover new patterns
        patterns = await ml_optimizer.discover_rule_patterns()
        
        console.print(f"[green]Discovered {len(patterns)} new patterns[/green]")
        for pattern in patterns:
            console.print(f"  - {pattern['pattern_name']}: {pattern['effectiveness_score']:.2f}")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
    
    # Optimize existing rules
    rule_ids = ['clarity_rule', 'specificity_rule']  # Get from config
    for rule_id in rule_ids:
        # Get performance data for the rule
        performance_data = await ml_optimizer.get_rule_performance_data(rule_id)
        if performance_data:
            # Call the actual RuleOptimizer.optimize_rule method with proper parameters
            optimization_result = await ml_optimizer.rule_optimizer.optimize_rule(
                rule_id=rule_id,
                performance_data=performance_data,
                historical_data=await ml_optimizer.get_historical_data(rule_id)
            )
            console.print(f"[green]Optimized {rule_id}: {optimization_result['status']}[/green]")
        else:
            console.print(f"[yellow]Insufficient data for {rule_id}[/yellow]")
```

### Phase 3: MCP Server Integration

**For External Tool Access:**

```python
# src/prompt_improver/mcp_server/enhanced_api.py
from mcp import Server
from ..database import HybridDatabaseClient, DatabaseConfig
import json

app = Server("apes-enhanced")
db_config = DatabaseConfig()
db_client = HybridDatabaseClient(db_config.postgres_url)

@app.list_tools()
async def list_tools():
    """List available MCP tools"""
    return [
        {
            "name": "improve_prompt",
            "description": "Improve a prompt using data-driven rule selection",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "user_context": {"type": "object"},
                    "session_id": {"type": "string"}
                },
                "required": ["prompt"]
            }
        },
        {
            "name": "get_rule_analytics",
            "description": "Get rule effectiveness analytics",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "default": 30}
                }
            }
        },
        {
            "name": "query_database",
            "description": "Execute analytical queries on the APES database",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "parameters": {"type": "array"}
                },
                "required": ["query"]
            }
        }
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls"""
    if name == "improve_prompt":
        # Route to rule engine
        result = await rule_engine.improve_prompt(
            prompt=arguments["prompt"],
            user_context=arguments.get("user_context"),
            session_id=arguments.get("session_id")
        )
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    
    elif name == "get_rule_analytics":
        # Route to ML optimizer
        analytics = await ml_optimizer.analyze_rule_effectiveness(
            days=arguments.get("days", 30)
        )
        return {"content": [{"type": "text", "text": json.dumps(analytics, indent=2)}]}
    
    elif name == "query_database":
        # Execute analytical query
        if db_client.direct_client.pool:
            async with db_client.direct_client.pool.acquire() as conn:
                rows = await conn.fetch(
                    arguments["query"],
                    *arguments.get("parameters", [])
                )
                result = [dict(row) for row in rows]
                return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        else:
            return {"content": [{"type": "text", "text": "Database connection not available"}]}
    
    else:
        raise ValueError(f"Unknown tool: {name}")
```

## Implementation Timeline

### Week 1: Database Integration Foundation
- ✅ **Day 1-2:** PostgreSQL setup (COMPLETED)
- **Day 3-4:** Implement database client classes (Direct, MCP, Hybrid)
- **Day 5-7:** Update configuration management and testing

### Week 2: Rule Engine Enhancement
- **Day 1-3:** Implement enhanced rule engine with database integration
- **Day 4-5:** Update existing rules to work with new architecture
- **Day 6-7:** Performance testing and optimization

### Week 3: ML Optimizer Enhancement
- **Day 1-3:** Implement enhanced ML optimizer with analytics
- **Day 4-5:** Integrate with MLflow and add pattern discovery
- **Day 6-7:** Testing and validation

### Week 4: CLI and MCP Integration
- **Day 1-3:** Update CLI with enhanced database analytics
- **Day 4-5:** Implement enhanced MCP server capabilities
- **Day 6-7:** End-to-end testing and documentation

### Week 5: Production Deployment
- **Day 1-2:** Production configuration and security
- **Day 3-4:** Deployment and monitoring setup
- **Day 5-7:** User testing and feedback collection

## Expected Benefits

### Performance Improvements
- **50-70% better rule selection** through data-driven optimization
- **Real-time rule effectiveness** feedback loop
- **Automated rule parameter optimization** using historical data

### Analytics and Insights
- **Rule effectiveness dashboards** with trend analysis
- **User satisfaction correlation** with rule performance
- **A/B testing infrastructure** for rule combinations
- **Automated pattern discovery** for new rules

### Development Efficiency
- **Standardized external access** via MCP protocol
- **High-performance internal operations** via direct database access
- **Comprehensive analytics** for continuous improvement
- **Production-ready monitoring** and alerting

### User Experience
- **Smarter prompt improvements** based on proven effectiveness
- **Faster response times** through optimized rule selection
- **Continuous learning** from user feedback
- **Personalized recommendations** based on context and history

## Getting Started

1. **Start the database:**
   ```bash
   ./scripts/start_database.sh start
   ```

2. **Install dependencies:**
   ```bash
   pip install asyncpg sqlalchemy pydantic
   # APES unified MCP server - no external installation required
   ```

3. **Configure APES MCP server:**
   ```json
   {
     "mcpServers": {
       "apes-mcp": {
         "command": "python",
         "args": ["-m", "prompt_improver.mcp_server.server"],
         "cwd": "/path/to/prompt-improver",
         "env": {
           "PYTHONPATH": "/path/to/prompt-improver/src"
         }
       }
     }
   }
   ```

4. **Choose your implementation approach** and follow the specific code examples above.

## Recommendation: Hybrid Approach

For the APES project, I recommend **Option C: Hybrid Approach** because:

1. **Production Performance** - Direct database access for real-time rule selection
2. **External Integration** - MCP server for Claude Desktop and IDE integrations
3. **Future Flexibility** - Can optimize each path independently
4. **Best Practices** - Separates internal performance from external standardization
5. **Gradual Migration** - Can start with direct access and add MCP later

This approach will transform your static rule system into a dynamic, learning system that continuously improves based on real usage data and user feedback.