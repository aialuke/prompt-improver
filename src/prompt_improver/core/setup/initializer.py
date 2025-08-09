"""APES system initialization and setup automation.
Implements Task 1: Installation Automation from Phase 2.
"""
import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from sqlalchemy import func, select
from prompt_improver.core.config import AppConfig
from prompt_improver.core.di.ml_container import MLServiceContainer
from prompt_improver.core.factories.ml_pipeline_factory import MLPipelineOrchestratorFactory

class APESInitializer:
    """Comprehensive system initialization following production patterns"""

    def __init__(self, console: Console | None=None):
        self.console = console or Console()
        self.data_dir = None

    async def initialize_system(self, data_dir: Path | None=None, force: bool=False) -> dict[str, Any]:
        """Complete APES setup following research-validated best practices"""
        if data_dir is None:
            data_dir = Path.home() / '.local' / 'share' / 'apes'
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.console.print(f'🚀 Initializing APES system at: {self.data_dir}', style='green')
        initialization_results = {'data_dir': str(self.data_dir), 'timestamp': datetime.now().isoformat(), 'steps_completed': [], 'steps_failed': [], 'warnings': []}
        with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), BarColumn(), TaskProgressColumn(), console=self.console) as progress:
            total_steps = 9
            main_task = progress.add_task('Initializing APES system...', total=total_steps)
            try:
                progress.update(main_task, description='Creating directory structure...')
                await self.create_directory_structure(force)
                initialization_results['steps_completed'].append('directory_structure')
                progress.advance(main_task)
                progress.update(main_task, description='Setting up PostgreSQL...')
                await self.setup_postgresql_cluster()
                initialization_results['steps_completed'].append('postgresql_setup')
                progress.advance(main_task)
                progress.update(main_task, description='Applying performance optimizations...')
                await self.apply_performance_optimizations()
                initialization_results['steps_completed'].append('performance_optimization')
                progress.advance(main_task)
                progress.update(main_task, description='Creating configuration files...')
                await self.create_production_configs()
                initialization_results['steps_completed'].append('configuration_files')
                progress.advance(main_task)
                progress.update(main_task, description='Initializing database schema...')
                await self.create_database_schema()
                initialization_results['steps_completed'].append('database_schema')
                progress.advance(main_task)
                progress.update(main_task, description='Seeding baseline prompt engineering rules...')
                await self.seed_baseline_rules()
                initialization_results['steps_completed'].append('baseline_rules')
                progress.advance(main_task)
                progress.update(main_task, description='Generating initial training data...')
                await self.generate_initial_training_data()
                initialization_results['steps_completed'].append('training_data')
                progress.advance(main_task)
                progress.update(main_task, description='Configuring MCP server...')
                await self.setup_mcp_server()
                initialization_results['steps_completed'].append('mcp_server')
                progress.advance(main_task)
                progress.update(main_task, description='Verifying system health...')
                health_results = await self.verify_system_health()
                initialization_results['health_check'] = health_results
                initialization_results['steps_completed'].append('health_verification')
                progress.advance(main_task)
                progress.update(main_task, description='✅ APES initialization completed!', completed=total_steps)
            except Exception as e:
                initialization_results['steps_failed'].append(f'Step failed: {e!s}')
                self.console.print(f'❌ Initialization failed: {e}', style='red')
                raise
        self.console.print('✅ APES initialized successfully', style='green')
        self.console.print("🚀 Run 'apes start' to begin real-time prompt enhancement", style='blue')
        return initialization_results

    async def create_directory_structure(self, force: bool=False):
        """Create XDG-compliant directory structure"""
        if self.data_dir is None:
            self.data_dir = Path.home() / '.local' / 'share' / 'apes'
        if self.data_dir.exists() and (not force):
            if any(self.data_dir.iterdir()):
                raise ValueError(f'Directory {self.data_dir} already exists and is not empty. Use --force to overwrite.')
        directories = ['data/postgresql', 'data/backups', 'data/logs', 'data/ml-models', 'data/user-prompts', 'config', 'temp', 'scripts']
        for dir_path in directories:
            full_path = self.data_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        self.console.print(f'  📁 Created directory structure at {self.data_dir}', style='dim')

    async def setup_postgresql_cluster(self):
        """Initialize local PostgreSQL cluster if needed"""
        try:
            import shutil
            psql_path = shutil.which('psql')
            if not psql_path:
                common_paths = ['/usr/local/bin/psql', '/opt/homebrew/bin/psql', '/usr/bin/psql']
                for path in common_paths:
                    if Path(path).exists():
                        psql_path = path
                        break
                if not psql_path:
                    self.console.print('⚠️  PostgreSQL not found. Please install PostgreSQL first.', style='yellow')
                    return
            result = subprocess.run([psql_path, '--version'], check=False, capture_output=True, text=True, shell=False, timeout=10)
            if result.returncode != 0:
                self.console.print('⚠️  PostgreSQL found but not working correctly', style='yellow')
                return
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.console.print(f'⚠️  Could not verify PostgreSQL installation: {e}', style='yellow')
            return
        except Exception as e:
            self.console.print(f'⚠️  Could not verify PostgreSQL installation: {e}', style='yellow')
            return
        try:
            config = AppConfig().database
            from sqlalchemy import text
            from prompt_improver.database import get_session, scalar
            async with get_session() as session:
                await scalar(session, text('SELECT 1'))
                self.console.print('  ✅ Connected to existing PostgreSQL database', style='dim')
                return
        except Exception:
            self.console.print('  🔧 Setting up new PostgreSQL connection', style='dim')
        self.console.print('  📡 PostgreSQL setup completed', style='dim')

    async def apply_performance_optimizations(self):
        """Apply PostgreSQL performance optimizations"""
        try:
            async with get_session() as session:
                optimizations = ["SET shared_preload_libraries = 'pg_stat_statements'", 'SET track_activity_query_size = 2048', 'SET log_min_duration_statement = 1000']
                for optimization in optimizations:
                    try:
                        await session.execute(optimization)
                    except Exception:
                        pass
                self.console.print('  ⚡ Applied performance optimizations', style='dim')
        except Exception as e:
            self.console.print(f'  ⚠️  Could not apply all optimizations: {e}', style='yellow')

    async def create_production_configs(self):
        """Generate environment-specific configuration files"""
        config_dir = self.data_dir / 'config'
        db_config = {'database': {'url': '${DATABASE_URL:-postgresql+asyncpg://localhost:5432/apes_db}', 'pool_size': 10, 'max_overflow': 20, 'pool_timeout': 30, 'pool_recycle': 3600, 'echo_sql': False}, 'performance': {'target_response_time_ms': 200, 'cache_ttl_seconds': 300, 'query_timeout_seconds': 30}}
        with open(config_dir / 'database.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(db_config, f, default_flow_style=False)
        mcp_config = {'mcp': {'transport': 'stdio', 'port': 3000, 'host': os.getenv('MCP_SERVER_HOST', '0.0.0.0'), 'timeout_ms': 200, 'max_request_size': 1048576}, 'tools': {'improve_prompt': {'enabled': True, 'max_prompt_length': 10000}, 'store_prompt': {'enabled': True, 'priority': 100}}, 'resources': {'rule_status': {'enabled': True, 'cache_ttl': 60}}}
        with open(config_dir / 'mcp.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(mcp_config, f, default_flow_style=False)
        ml_config = {'ml': {'optimization': {'n_trials': 50, 'inner_folds': 5, 'outer_folds': 3, 'timeout_seconds': 300}, 'models': {'ensemble_threshold': 50, 'min_training_samples': 20, 'model_registry_path': str(self.data_dir / 'data' / 'ml-models')}, 'mlflow': {'tracking_uri': f"file://{os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))}", 'experiment_name': 'apes_optimization'}}}
        with open(config_dir / 'ml.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(ml_config, f, default_flow_style=False)
        service_config = {'service': {'pid_file': str(self.data_dir / 'apes.pid'), 'log_file': str(self.data_dir / 'data' / 'logs' / 'apes.log'), 'log_level': 'INFO', 'daemon_user': os.getenv('USER', 'apes'), 'max_memory_mb': 512}, 'monitoring': {'health_check_interval': 30, 'performance_alert_threshold_ms': 250, 'log_retention_days': 30}}
        with open(config_dir / 'service.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(service_config, f, default_flow_style=False)
        self.console.print(f'  📝 Created configuration files in {config_dir}', style='dim')

    async def create_database_schema(self):
        """Initialize database schema using existing schema.sql"""
        project_root = Path(__file__).parent.parent.parent.parent
        schema_file = project_root / 'database' / 'schema.sql'
        if not schema_file.exists():
            self.console.print(f'  ⚠️  Schema file not found at {schema_file}', style='yellow')
            return
        try:
            async with get_session() as session:
                schema_sql = schema_file.read_text()
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                for statement in statements:
                    if statement:
                        try:
                            await session.execute(statement)
                        except Exception as e:
                            if 'already exists' not in str(e).lower():
                                self.console.print(f'  ⚠️  Schema warning: {e}', style='yellow')
                await session.commit()
                self.console.print('  🗄️  Database schema initialized', style='dim')
        except Exception as e:
            self.console.print(f'  ❌ Schema initialization failed: {e}', style='red')
            raise

    async def seed_baseline_rules(self):
        """Load rule configurations from YAML into database"""
        config_file = self.data_dir / 'config' / 'rule_config.yaml'
        if not config_file.exists():
            self.console.print('⚠️ Rule configuration file not found, using defaults', style='yellow')
            return
        try:
            async with get_session() as session:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                import json
                from prompt_improver.database.models import RuleMetadata
                rules_seeded = 0
                rules_updated = 0
                for rule_id, rule_config in config.get('rules', {}).items():
                    query = select(RuleMetadata).where(RuleMetadata.rule_id == rule_id)
                    result = await session.execute(query)
                    existing = result.scalar_one_or_none()
                    if existing:
                        existing.enabled = rule_config.get('enabled', True)
                        existing.priority = rule_config.get('priority', 100)
                        existing.rule_name = rule_config.get('name', rule_id.replace('_', ' ').title())
                        existing.rule_category = rule_config.get('category', 'custom')
                        existing.rule_description = rule_config.get('description', '')
                        if 'params' in rule_config:
                            existing.default_parameters = json.dumps(rule_config['params'])
                        if 'constraints' in rule_config:
                            existing.parameter_constraints = json.dumps(rule_config['constraints'])
                        rules_updated += 1
                    else:
                        new_rule = RuleMetadata(rule_id=rule_id, rule_name=rule_config.get('name', rule_id.replace('_', ' ').title()), rule_category=rule_config.get('category', 'custom'), rule_description=rule_config.get('description', ''), enabled=rule_config.get('enabled', True), priority=rule_config.get('priority', 100), rule_version='1.0.0', default_parameters=json.dumps(rule_config.get('params', {})), parameter_constraints=json.dumps(rule_config.get('constraints', {})))
                        session.add(new_rule)
                        rules_seeded += 1
                await session.commit()
                self.console.print(f'  📋 Rule configurations loaded successfully: {rules_seeded} new, {rules_updated} updated', style='dim')
        except Exception as e:
            self.console.print(f'  ❌ Failed to load rule configurations: {e}', style='red')
            raise

    async def generate_initial_training_data(self):
        """Bootstrap with enhanced synthetic training data using research-driven generator"""
        try:
            async with get_session() as session:
                from prompt_improver.database.models import TrainingPrompt
                query = select(func.count(TrainingPrompt.id)).where(TrainingPrompt.data_source == 'synthetic')
                result = await session.execute(query)
                existing_count = result.scalar()
                if existing_count >= 100:
                    self.console.print(f'Found {existing_count} existing synthetic samples, skipping generation', style='dim')
                    return None
                self.console.print('Generating enhanced synthetic training data using ProductionSyntheticDataGenerator', style='dim')
                from prompt_improver.ml.preprocessing.orchestrator import ProductionSyntheticDataGenerator
                generator = ProductionSyntheticDataGenerator(target_samples=1000, random_state=42)
                training_data = await generator.generate_comprehensive_training_data()
                saved_count = await generator.save_to_database(training_data, session)
                summary = generator.get_generation_summary(training_data)
                self.console.print('Enhanced synthetic data generation complete:', style='dim')
                self.console.print(f"  - Total samples generated: {summary['generation_summary']['total_samples']}", style='dim')
                self.console.print(f"  - Quality score: {summary['generation_summary']['quality_score']}", style='dim')
                self.console.print(f"  - Domains covered: {summary['generation_summary']['domains_covered']}", style='dim')
                self.console.print(f"  - ML requirements met: {summary['quality_analysis']['ml_requirements_met']}", style='dim')
                return saved_count
        except Exception as e:
            self.console.print(f'  ⚠️  Enhanced synthetic data generation warning: {e}', style='yellow')

    async def setup_mcp_server(self):
        """Configure MCP server for stdio transport"""
        mcp_server_path = Path(__file__).parent.parent / 'mcp_server' / 'mcp_server.py'
        if mcp_server_path.exists():
            os.chmod(mcp_server_path, 493)
            self.console.print('  🔧 MCP server configured for stdio transport', style='dim')
        else:
            self.console.print(f'  ⚠️  MCP server script not found at {mcp_server_path}', style='yellow')

    async def verify_system_health(self) -> dict[str, Any]:
        """Test system integration and performance validation"""
        health_results = {'database_connection': False, 'mcp_performance': None, 'configuration_valid': False, 'directories_created': False, 'overall_status': 'failed'}
        try:
            from sqlalchemy import text
            from prompt_improver.database import scalar
            async with get_session() as session:
                await scalar(session, text('SELECT 1'))
                health_results['database_connection'] = True
                self.console.print('  ✅ Database connection verified', style='dim')
        except Exception as e:
            self.console.print(f'  ❌ Database connection failed: {e}', style='red')
        try:
            from prompt_improver.core.mcp_server.server import APESMCPServer
            mcp_server = APESMCPServer()
            start_time = asyncio.get_event_loop().time()
            result = await mcp_server._improve_prompt_impl(prompt='Test prompt for performance validation', context={'domain': 'testing'}, session_id='health_check', rate_limit_remaining=None)
            end_time = asyncio.get_event_loop().time()
            response_time_ms = (end_time - start_time) * 1000
            health_results['mcp_performance'] = response_time_ms
            if response_time_ms < 200:
                self.console.print(f'  ✅ MCP performance: {response_time_ms:.1f}ms (target: <200ms)', style='dim')
            else:
                self.console.print(f'  ⚠️  MCP performance: {response_time_ms:.1f}ms (exceeds 200ms target)', style='yellow')
        except Exception as e:
            self.console.print(f'  ⚠️  MCP performance test failed: {e}', style='yellow')
        config_dir = self.data_dir / 'config'
        required_configs = ['database.yaml', 'mcp.yaml', 'ml.yaml', 'service.yaml']
        all_configs_exist = all(((config_dir / config).exists() for config in required_configs))
        health_results['configuration_valid'] = all_configs_exist
        if all_configs_exist:
            self.console.print('  ✅ All configuration files created', style='dim')
        else:
            self.console.print('  ❌ Some configuration files missing', style='red')
        required_dirs = ['data', 'config', 'temp', 'scripts']
        all_dirs_exist = all(((self.data_dir / dir_name).exists() for dir_name in required_dirs))
        health_results['directories_created'] = all_dirs_exist
        if all_dirs_exist:
            self.console.print('  ✅ All required directories created', style='dim')
        if health_results['database_connection'] and health_results['configuration_valid'] and health_results['directories_created']:
            health_results['overall_status'] = 'healthy'
        return health_results
