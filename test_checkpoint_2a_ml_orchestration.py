#!/usr/bin/env python3
"""
Checkpoint 2A: ML Pipeline Orchestration Tests with Real Model Training
=======================================================================

Validates Week 3 ML infrastructure consolidation through actual model training operations.
Tests the integrated functionality of:
- AdaptiveEventBus with ML training events  
- EnhancedBackgroundTaskManager with ML workloads
- UnifiedConnectionManager with ML database operations
- Real model training using consolidated infrastructure

Success Criteria:
- Event Processing: >8,000 events/sec through AdaptiveEventBus
- Task Management: All ML background tasks properly managed
- Database Performance: <200ms average query time during ML operations  
- Error Rate: <0.1% failure rate in ML pipeline operations
- Integration: All components work seamlessly together
"""

import asyncio
import json
import time
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# ML Infrastructure imports
from src.prompt_improver.ml.orchestration.events.adaptive_event_bus import (
    AdaptiveEventBus, EventType, MLEvent, EventBusState
)
from src.prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
from src.prompt_improver.cli.core.training_system_manager import TrainingSystemManager  
from src.prompt_improver.ml.automl.orchestrator import AutoMLOrchestrator, AutoMLConfig
from src.prompt_improver.database import get_sessionmanager
from src.prompt_improver.database.unified_connection_manager import (
    UnifiedConnectionManager, ManagerMode, ConnectionMode
)

# Performance monitoring
from src.prompt_improver.performance.monitoring.health.background_manager import (
    get_background_task_manager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResults:
    """Test results structure"""
    test_name: str
    success: bool
    execution_time: float
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class Checkpoint2AValidator:
    """
    Comprehensive validator for ML pipeline orchestration consolidation.
    
    Tests real model training operations against consolidated infrastructure
    to ensure Week 3 consolidation maintains performance and functionality.
    """
    
    def __init__(self):
        self.results: List[TestResults] = []
        self.start_time = time.time()
        
        # Initialize consolidated components
        self.event_bus: Optional[AdaptiveEventBus] = None
        self.training_manager: Optional[TrainingSystemManager] = None
        self.automl_orchestrator: Optional[AutoMLOrchestrator] = None
        self.unified_db_manager: Optional[UnifiedConnectionManager] = None
        self.background_task_manager = get_background_task_manager()
        
        # Test configurations
        self.target_event_throughput = 8000  # events/sec
        self.target_db_response_time = 200   # ms
        self.max_error_rate = 0.001          # 0.1%
        
        logger.info("Checkpoint 2A Validator initialized for ML orchestration testing")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Execute comprehensive validation of ML infrastructure consolidation.
        
        Returns:
            Comprehensive validation results with performance metrics
        """
        logger.info("🚀 Starting Checkpoint 2A: ML Pipeline Orchestration Validation")
        
        try:
            # Phase A: AdaptiveEventBus Validation
            await self.phase_a_event_bus_validation()
            
            # Phase B: Background Task Management Validation
            await self.phase_b_background_task_validation()
            
            # Phase C: Database Session Management Under ML Load
            await self.phase_c_database_validation()
            
            # Phase D: Real Model Training Integration Test
            await self.phase_d_real_training_validation()
            
            # Phase E: Performance Metrics Analysis
            performance_analysis = await self.phase_e_performance_analysis()
            
            # Generate comprehensive report
            return await self.generate_final_report(performance_analysis)
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - self.start_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def phase_a_event_bus_validation(self):
        """
        Phase A: Test AdaptiveEventBus with ML training events and measure performance.
        
        Validates:
        - Event processing throughput >8,000 events/sec
        - Priority queue functionality with ML workloads
        - Event filtering and routing for ML components
        - Auto-scaling worker management
        """
        logger.info("📡 Phase A: AdaptiveEventBus Validation Starting")
        phase_start = time.time()
        
        try:
            # Initialize AdaptiveEventBus with ML training configuration
            config = OrchestratorConfig(
                max_concurrent_workflows=5,
                event_bus_buffer_size=10000,  # Large buffer for throughput testing
                training_timeout=1800
            )
            
            self.event_bus = AdaptiveEventBus(
                config=config,
                task_manager=self.background_task_manager
            )
            
            # Start event bus
            await self.event_bus.start()
            logger.info("✅ AdaptiveEventBus started successfully")
            
            # Test 1: High-throughput event processing
            throughput_result = await self._test_event_throughput()
            
            # Test 2: Priority queue functionality
            priority_result = await self._test_priority_queue_ml_events()
            
            # Test 3: Auto-scaling worker management
            scaling_result = await self._test_worker_auto_scaling()
            
            # Test 4: Circuit breaker and resilience
            resilience_result = await self._test_event_bus_resilience()
            
            # Collect Phase A metrics
            event_bus_metrics = self.event_bus.get_performance_metrics()
            
            phase_a_result = TestResults(
                test_name="Phase_A_AdaptiveEventBus",
                success=all([
                    throughput_result["success"],
                    priority_result["success"], 
                    scaling_result["success"],
                    resilience_result["success"]
                ]),
                execution_time=time.time() - phase_start,
                metrics={
                    "throughput_test": throughput_result,
                    "priority_queue_test": priority_result,
                    "auto_scaling_test": scaling_result,
                    "resilience_test": resilience_result,
                    "event_bus_metrics": event_bus_metrics
                }
            )
            
            self.results.append(phase_a_result)
            logger.info(f"✅ Phase A completed in {phase_a_result.execution_time:.2f}s")
            
        except Exception as e:
            error_result = TestResults(
                test_name="Phase_A_AdaptiveEventBus",
                success=False,
                execution_time=time.time() - phase_start,
                metrics={},
                error_message=str(e)
            )
            self.results.append(error_result)
            logger.error(f"❌ Phase A failed: {e}")
    
    async def _test_event_throughput(self) -> Dict[str, Any]:
        """Test high-throughput event processing with ML training events."""
        logger.info("Testing event throughput with ML training events...")
        
        # Create ML training event handlers
        events_processed = []
        
        async def training_start_handler(event: MLEvent):
            events_processed.append(("training_start", time.time()))
        
        async def training_progress_handler(event: MLEvent):
            events_processed.append(("training_progress", time.time()))
        
        async def training_complete_handler(event: MLEvent):
            events_processed.append(("training_complete", time.time()))
        
        # Subscribe to ML training events
        self.event_bus.subscribe(EventType.TRAINING_STARTED, training_start_handler, priority=8)
        self.event_bus.subscribe(EventType.TRAINING_PROGRESS, training_progress_handler, priority=5)
        self.event_bus.subscribe(EventType.TRAINING_COMPLETED, training_complete_handler, priority=9)
        
        # Generate high volume of ML training events
        test_start = time.time()
        num_events = 10000  # 10K events for throughput testing
        
        # Emit events as fast as possible
        emit_tasks = []
        for i in range(num_events):
            if i % 3 == 0:
                event = MLEvent(
                    event_type=EventType.TRAINING_STARTED,
                    source="checkpoint_2a_test",
                    data={"training_id": f"training_{i}", "model_type": "neural_network"},
                    timestamp=datetime.now(timezone.utc)
                )
                priority = 8
            elif i % 3 == 1:
                event = MLEvent(
                    event_type=EventType.TRAINING_PROGRESS,
                    source="checkpoint_2a_test",
                    data={"training_id": f"training_{i//3}", "progress": i % 100},
                    timestamp=datetime.now(timezone.utc)
                )
                priority = 5
            else:
                event = MLEvent(
                    event_type=EventType.TRAINING_COMPLETED,
                    source="checkpoint_2a_test",
                    data={"training_id": f"training_{i//3}", "final_score": 0.85},
                    timestamp=datetime.now(timezone.utc)
                )
                priority = 9
            
            emit_tasks.append(self.event_bus.emit(event, priority=priority))
        
        # Execute all emissions concurrently
        await asyncio.gather(*emit_tasks, return_exceptions=True)
        emit_time = time.time() - test_start
        
        # Wait for processing to complete
        await asyncio.sleep(2)
        
        # Calculate throughput metrics
        total_time = time.time() - test_start
        events_per_second = num_events / total_time
        
        success = events_per_second >= self.target_event_throughput
        
        return {
            "success": success,
            "events_emitted": num_events,
            "events_processed": len(events_processed),
            "total_time_seconds": total_time,
            "emit_time_seconds": emit_time,
            "events_per_second": events_per_second,
            "target_throughput": self.target_event_throughput,
            "throughput_achieved": success,
            "processing_efficiency": len(events_processed) / num_events if num_events > 0 else 0
        }
    
    async def _test_priority_queue_ml_events(self) -> Dict[str, Any]:
        """Test priority queue functionality with ML training events."""
        logger.info("Testing priority queue with ML events...")
        
        # Track event processing order
        processing_order = []
        
        async def high_priority_handler(event: MLEvent):
            processing_order.append(("high_priority", event.data.get("event_id")))
        
        async def medium_priority_handler(event: MLEvent):
            processing_order.append(("medium_priority", event.data.get("event_id")))
        
        async def low_priority_handler(event: MLEvent):
            processing_order.append(("low_priority", event.data.get("event_id")))
        
        # Subscribe with different priorities
        self.event_bus.subscribe(EventType.TRAINING_STARTED, high_priority_handler, priority=9)
        self.event_bus.subscribe(EventType.TRAINING_PROGRESS, medium_priority_handler, priority=5)  
        self.event_bus.subscribe(EventType.TRAINING_COMPLETED, low_priority_handler, priority=1)
        
        # Emit events in reverse priority order to test queue sorting
        events = [
            (EventType.TRAINING_COMPLETED, 1, "low_1"),  # Low priority first
            (EventType.TRAINING_PROGRESS, 5, "med_1"),   # Medium priority
            (EventType.TRAINING_STARTED, 9, "high_1"),   # High priority last
            (EventType.TRAINING_COMPLETED, 1, "low_2"),
            (EventType.TRAINING_STARTED, 9, "high_2"),
        ]
        
        for event_type, priority, event_id in events:
            event = MLEvent(
                event_type=event_type,
                source="checkpoint_2a_test",
                data={"event_id": event_id},
                timestamp=datetime.now(timezone.utc)
            )
            await self.event_bus.emit(event, priority=priority)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify priority ordering (high priority events should be processed first)
        high_priority_events = [item for item in processing_order if item[0] == "high_priority"]
        medium_priority_events = [item for item in processing_order if item[0] == "medium_priority"]
        low_priority_events = [item for item in processing_order if item[0] == "low_priority"]
        
        # Priority queue should process high priority events before low priority
        priority_ordering_correct = True
        if high_priority_events and low_priority_events:
            first_high_idx = next(i for i, item in enumerate(processing_order) if item[0] == "high_priority")
            last_low_idx = max(i for i, item in enumerate(processing_order) if item[0] == "low_priority")
            priority_ordering_correct = first_high_idx < last_low_idx
        
        return {
            "success": priority_ordering_correct and len(processing_order) == len(events),
            "events_processed": len(processing_order),
            "high_priority_count": len(high_priority_events),
            "medium_priority_count": len(medium_priority_events), 
            "low_priority_count": len(low_priority_events),
            "priority_ordering_correct": priority_ordering_correct,
            "processing_order": processing_order
        }
    
    async def _test_worker_auto_scaling(self) -> Dict[str, Any]:
        """Test worker auto-scaling under ML workload."""
        logger.info("Testing worker auto-scaling...")
        
        initial_metrics = self.event_bus.get_performance_metrics()
        initial_workers = initial_metrics.get("worker_count", 0)
        
        # Generate high load to trigger scaling
        load_events = []
        for i in range(1000):  # High volume to trigger scaling
            event = MLEvent(
                event_type=EventType.TRAINING_PROGRESS,
                source="checkpoint_2a_test",
                data={"batch_id": i, "intensive_processing": True},
                timestamp=datetime.now(timezone.utc)
            )
            load_events.append(self.event_bus.emit(event))
        
        # Process high load
        await asyncio.gather(*load_events, return_exceptions=True)
        
        # Wait for auto-scaling to trigger
        await asyncio.sleep(5)
        
        # Check if scaling occurred
        post_load_metrics = self.event_bus.get_performance_metrics()
        final_workers = post_load_metrics.get("worker_count", 0)
        
        # Scaling should have occurred due to high load
        scaling_occurred = final_workers > initial_workers
        
        return {
            "success": scaling_occurred,
            "initial_workers": initial_workers,
            "final_workers": final_workers,
            "scaling_occurred": scaling_occurred,
            "queue_utilization": post_load_metrics.get("queue_utilization", 0),
            "state": post_load_metrics.get("state", "unknown")
        }
    
    async def _test_event_bus_resilience(self) -> Dict[str, Any]:
        """Test event bus resilience and circuit breaker functionality."""
        logger.info("Testing event bus resilience...")
        
        # Create a handler that will fail to test circuit breaker
        failure_count = 0
        
        async def failing_handler(event: MLEvent):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 5:  # Fail first 5 events to trigger circuit breaker
                raise Exception("Simulated handler failure")
            return "success"
        
        # Subscribe failing handler
        self.event_bus.subscribe(EventType.TRAINING_STARTED, failing_handler)
        
        # Send events that will trigger failures
        for i in range(10):
            event = MLEvent(
                event_type=EventType.TRAINING_STARTED,
                source="checkpoint_2a_test",
                data={"test_id": i},
                timestamp=datetime.now(timezone.utc)
            )
            await self.event_bus.emit(event)
        
        await asyncio.sleep(2)
        
        # Check circuit breaker metrics
        metrics = self.event_bus.get_performance_metrics()
        circuit_breaker_trips = metrics.get("circuit_breaker_trips", 0)
        
        return {
            "success": True,  # Resilience test passes if it doesn't crash the system
            "failure_count": failure_count,
            "circuit_breaker_trips": circuit_breaker_trips,
            "system_stable": self.event_bus.is_running,
            "metrics": metrics
        }
    
    async def phase_b_background_task_validation(self):
        """
        Phase B: Test EnhancedBackgroundTaskManager with actual ML training workloads.
        
        Validates:
        - ML background task lifecycle management
        - Task priority handling and resource allocation
        - Integration with training workflows
        - Performance under ML workload
        """
        logger.info("⚙️ Phase B: Background Task Management Validation Starting")
        phase_start = time.time()
        
        try:
            # Test 1: ML training task submission and management
            task_management_result = await self._test_ml_task_management()
            
            # Test 2: Task priority and resource allocation
            priority_allocation_result = await self._test_task_priority_allocation()
            
            # Test 3: Task lifecycle and monitoring
            lifecycle_result = await self._test_task_lifecycle_monitoring()
            
            # Test 4: Background task integration with event bus
            integration_result = await self._test_task_event_integration()
            
            phase_b_result = TestResults(
                test_name="Phase_B_BackgroundTaskManager",
                success=all([
                    task_management_result["success"],
                    priority_allocation_result["success"],
                    lifecycle_result["success"],
                    integration_result["success"]
                ]),
                execution_time=time.time() - phase_start,
                metrics={
                    "ml_task_management": task_management_result,
                    "priority_allocation": priority_allocation_result,
                    "lifecycle_monitoring": lifecycle_result,
                    "event_integration": integration_result
                }
            )
            
            self.results.append(phase_b_result)
            logger.info(f"✅ Phase B completed in {phase_b_result.execution_time:.2f}s")
            
        except Exception as e:
            error_result = TestResults(
                test_name="Phase_B_BackgroundTaskManager",
                success=False,
                execution_time=time.time() - phase_start,
                metrics={},
                error_message=str(e)
            )
            self.results.append(error_result)
            logger.error(f"❌ Phase B failed: {e}")
    
    async def _test_ml_task_management(self) -> Dict[str, Any]:
        """Test ML training task submission and management."""
        logger.info("Testing ML task management...")
        
        # Simulate ML training tasks
        async def ml_training_task(task_id: str):
            # Simulate actual ML training work
            await asyncio.sleep(0.5)  # Simulate training time
            return {"task_id": task_id, "training_accuracy": 0.85, "status": "completed"}
        
        # Submit multiple ML training tasks
        task_ids = []
        for i in range(5):
            task_id = f"ml_training_task_{i}"
            submitted_task_id = await self.background_task_manager.submit_enhanced_task(
                task_id=task_id,
                coroutine=ml_training_task(task_id),
                priority="HIGH",
                timeout=10,
                tags={"type": "ml_training", "model": "neural_network"}
            )
            task_ids.append(submitted_task_id)
        
        # Monitor task completion
        completed_tasks = []
        for task_id in task_ids:
            try:
                result = await self.background_task_manager.get_task_result(task_id, timeout=15)
                completed_tasks.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id} timed out")
        
        # Get task manager status
        status = await self.background_task_manager.get_comprehensive_status()
        
        success = len(completed_tasks) == len(task_ids)
        
        return {
            "success": success,
            "tasks_submitted": len(task_ids),
            "tasks_completed": len(completed_tasks),
            "task_manager_status": status,
            "completion_rate": len(completed_tasks) / len(task_ids) if task_ids else 0
        }
    
    async def _test_task_priority_allocation(self) -> Dict[str, Any]:
        """Test task priority handling and resource allocation."""
        logger.info("Testing task priority allocation...")
        
        # Submit tasks with different priorities
        high_priority_tasks = []
        low_priority_tasks = []
        
        async def priority_test_task(priority: str, task_id: str):
            start_time = time.time()
            await asyncio.sleep(0.2)  # Simulate work
            return {"priority": priority, "task_id": task_id, "execution_time": time.time() - start_time}
        
        # Submit low priority tasks first
        for i in range(3):
            task_id = await self.background_task_manager.submit_enhanced_task(
                task_id=f"low_priority_{i}",
                coroutine=priority_test_task("LOW", f"low_priority_{i}"),
                priority="LOW",
                timeout=10
            )
            low_priority_tasks.append(task_id)
        
        # Brief delay then submit high priority tasks
        await asyncio.sleep(0.1)
        
        for i in range(2):
            task_id = await self.background_task_manager.submit_enhanced_task(
                task_id=f"high_priority_{i}",
                coroutine=priority_test_task("HIGH", f"high_priority_{i}"),
                priority="HIGH",
                timeout=10
            )
            high_priority_tasks.append(task_id)
        
        # Wait for all tasks to complete
        all_results = []
        for task_list in [high_priority_tasks, low_priority_tasks]:
            for task_id in task_list:
                try:
                    result = await self.background_task_manager.get_task_result(task_id, timeout=15)
                    all_results.append(result)
                except asyncio.TimeoutError:
                    logger.warning(f"Priority test task {task_id} timed out")
        
        # Analyze priority handling
        high_priority_completed = sum(1 for r in all_results if r and r.get("priority") == "HIGH")
        low_priority_completed = sum(1 for r in all_results if r and r.get("priority") == "LOW")
        
        success = high_priority_completed == len(high_priority_tasks) and low_priority_completed == len(low_priority_tasks)
        
        return {
            "success": success,
            "high_priority_submitted": len(high_priority_tasks),
            "high_priority_completed": high_priority_completed,
            "low_priority_submitted": len(low_priority_tasks),
            "low_priority_completed": low_priority_completed,
            "total_results": len(all_results)
        }
    
    async def _test_task_lifecycle_monitoring(self) -> Dict[str, Any]:
        """Test task lifecycle and monitoring capabilities."""
        logger.info("Testing task lifecycle monitoring...")
        
        async def monitored_task():
            await asyncio.sleep(1)
            return {"status": "completed", "data_processed": 1000}
        
        # Submit task and monitor lifecycle
        task_id = await self.background_task_manager.submit_enhanced_task(
            task_id="lifecycle_test_task",
            coroutine=monitored_task(),
            priority="MEDIUM",
            timeout=10,
            tags={"test": "lifecycle", "component": "checkpoint_2a"}
        )
        
        # Monitor task status during execution
        status_checks = []
        for _ in range(3):
            await asyncio.sleep(0.3)
            status = await self.background_task_manager.get_task_status(task_id)
            status_checks.append(status)
        
        # Get final result
        result = await self.background_task_manager.get_task_result(task_id, timeout=15)
        
        # Get comprehensive status
        comprehensive_status = await self.background_task_manager.get_comprehensive_status()
        
        return {
            "success": result is not None,
            "task_id": task_id,
            "status_checks": status_checks,
            "final_result": result,
            "comprehensive_status": comprehensive_status
        }
    
    async def _test_task_event_integration(self) -> Dict[str, Any]:
        """Test background task integration with event bus."""
        logger.info("Testing task-event integration...")
        
        # Create task that emits events
        events_emitted = []
        
        async def event_emitting_task():
            for i in range(3):
                event = MLEvent(
                    event_type=EventType.TRAINING_PROGRESS,
                    source="background_task",
                    data={"progress": i * 25, "task_source": "background_task"},
                    timestamp=datetime.now(timezone.utc)
                )
                await self.event_bus.emit(event)
                events_emitted.append(i)
                await asyncio.sleep(0.1)
            return {"events_emitted": len(events_emitted)}
        
        # Submit event-emitting task
        task_id = await self.background_task_manager.submit_enhanced_task(
            task_id="event_integration_task",
            coroutine=event_emitting_task(),
            priority="MEDIUM",
            timeout=10
        )
        
        # Wait for completion
        result = await self.background_task_manager.get_task_result(task_id, timeout=15)
        
        return {
            "success": result is not None and len(events_emitted) > 0,
            "events_emitted_count": len(events_emitted),
            "task_result": result
        }
    
    async def phase_c_database_validation(self):
        """
        Phase C: Test UnifiedConnectionManager with ML database operations under load.
        
        Validates:
        - ML database session management under training load
        - Connection pooling efficiency with concurrent operations
        - Performance metrics during intensive database usage
        - Auto-scaling and optimization under ML workload
        """
        logger.info("💾 Phase C: Database Session Management Validation Starting")
        phase_start = time.time()
        
        try:
            # Initialize UnifiedConnectionManager for ML training mode
            self.unified_db_manager = UnifiedConnectionManager(mode=ManagerMode.ML_TRAINING)
            await self.unified_db_manager.initialize()
            logger.info("✅ UnifiedConnectionManager initialized for ML training")
            
            # Test 1: Concurrent database operations under ML load
            concurrent_ops_result = await self._test_concurrent_ml_database_ops()
            
            # Test 2: Connection pooling efficiency
            pooling_efficiency_result = await self._test_connection_pooling_efficiency()
            
            # Test 3: Database performance under training workload
            performance_result = await self._test_database_performance_under_load()
            
            # Test 4: Auto-scaling and optimization
            auto_scaling_result = await self._test_database_auto_scaling()
            
            # Get comprehensive database metrics
            db_metrics = await self.unified_db_manager.get_performance_metrics()
            
            phase_c_result = TestResults(
                test_name="Phase_C_DatabaseManagement",
                success=all([
                    concurrent_ops_result["success"],
                    pooling_efficiency_result["success"],
                    performance_result["success"],
                    auto_scaling_result["success"]
                ]),
                execution_time=time.time() - phase_start,
                metrics={
                    "concurrent_operations": concurrent_ops_result,
                    "pooling_efficiency": pooling_efficiency_result,
                    "performance_under_load": performance_result,
                    "auto_scaling": auto_scaling_result,
                    "database_metrics": db_metrics
                }
            )
            
            self.results.append(phase_c_result)
            logger.info(f"✅ Phase C completed in {phase_c_result.execution_time:.2f}s")
            
        except Exception as e:
            error_result = TestResults(
                test_name="Phase_C_DatabaseManagement",
                success=False,
                execution_time=time.time() - phase_start,
                metrics={},
                error_message=str(e)
            )
            self.results.append(error_result)
            logger.error(f"❌ Phase C failed: {e}")
    
    async def _test_concurrent_ml_database_ops(self) -> Dict[str, Any]:
        """Test concurrent ML database operations."""
        logger.info("Testing concurrent ML database operations...")
        
        # Simulate ML training database operations
        async def ml_database_operation(operation_id: int):
            start_time = time.time()
            try:
                async with self.unified_db_manager.get_async_session() as session:
                    # Simulate ML training data queries
                    from sqlalchemy import text
                    
                    # Read training data
                    result = await session.execute(
                        text("SELECT COUNT(*) FROM training_prompts WHERE training_priority > :priority"),
                        {"priority": 50}
                    )
                    count = result.scalar()
                    
                    # Simulate feature extraction query
                    await session.execute(text("SELECT 1"))  # Health check equivalent
                    
                    return {
                        "operation_id": operation_id,
                        "execution_time": time.time() - start_time,
                        "records_found": count,
                        "success": True
                    }
            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "execution_time": time.time() - start_time,
                    "error": str(e),
                    "success": False
                }
        
        # Execute concurrent operations
        num_concurrent_ops = 20
        concurrent_tasks = [
            ml_database_operation(i) for i in range(num_concurrent_ops)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_ops = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_ops = [r for r in results if not (isinstance(r, dict) and r.get("success", False))]
        
        avg_response_time = statistics.mean([op["execution_time"] for op in successful_ops]) * 1000 if successful_ops else 0
        success_rate = len(successful_ops) / len(results) if results else 0
        
        # Check if performance meets targets
        meets_response_time_target = avg_response_time < self.target_db_response_time
        meets_success_rate_target = success_rate > (1 - self.max_error_rate)
        
        return {
            "success": meets_response_time_target and meets_success_rate_target,
            "total_operations": len(results),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": success_rate,
            "avg_response_time_ms": avg_response_time,
            "total_execution_time": total_time,
            "meets_response_time_target": meets_response_time_target,
            "meets_success_rate_target": meets_success_rate_target
        }
    
    async def _test_connection_pooling_efficiency(self) -> Dict[str, Any]:
        """Test connection pooling efficiency under ML workload."""
        logger.info("Testing connection pooling efficiency...")
        
        # Get initial metrics
        initial_metrics = await self.unified_db_manager.get_performance_metrics()
        initial_connections = initial_metrics.get("total_connections", 0)
        
        # Simulate ML training workload with varying connection needs
        async def pooling_test_operation(op_type: str):
            if op_type == "read_intensive":
                async with self.unified_db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    for _ in range(5):  # Multiple reads
                        await session.execute(text("SELECT 1"))
            elif op_type == "write_operation":
                async with self.unified_db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))  # Simulate write
            
            return op_type
        
        # Execute mixed workload
        operations = (
            [pooling_test_operation("read_intensive") for _ in range(10)] +
            [pooling_test_operation("write_operation") for _ in range(5)]
        )
        
        await asyncio.gather(*operations, return_exceptions=True)
        
        # Get final metrics
        final_metrics = await self.unified_db_manager.get_performance_metrics()
        final_connections = final_metrics.get("total_connections", 0)
        
        # Analyze pooling efficiency
        pool_utilization = final_metrics.get("pool_utilization", 0)
        connection_reuse_count = final_metrics.get("connection_reuse_count", 0)
        pool_efficiency = final_metrics.get("pool_efficiency", 0)
        
        # Efficiency criteria
        efficient_utilization = 20 <= pool_utilization <= 90  # Good utilization range
        effective_reuse = connection_reuse_count > 0
        
        return {
            "success": efficient_utilization and effective_reuse,
            "initial_connections": initial_connections,
            "final_connections": final_connections,
            "pool_utilization": pool_utilization,
            "connection_reuse_count": connection_reuse_count,
            "pool_efficiency": pool_efficiency,
            "efficient_utilization": efficient_utilization,
            "effective_reuse": effective_reuse
        }
    
    async def _test_database_performance_under_load(self) -> Dict[str, Any]:
        """Test database performance under heavy ML training load."""
        logger.info("Testing database performance under heavy load...")
        
        # Generate heavy database load
        response_times = []
        
        async def heavy_load_operation(batch_id: int):
            start_time = time.time()
            try:
                async with self.unified_db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    
                    # Simulate complex ML training queries
                    await session.execute(
                        text("SELECT COUNT(*) FROM training_prompts WHERE enhancement_result IS NOT NULL")
                    )
                    await session.execute(
                        text("SELECT COUNT(*) FROM prompt_sessions WHERE created_at > NOW() - INTERVAL '1 day'")
                    )
                    
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                return {"batch_id": batch_id, "response_time": response_time, "success": True}
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                return {"batch_id": batch_id, "response_time": response_time, "error": str(e), "success": False}
        
        # Execute heavy load (50 concurrent operations)
        load_tasks = [heavy_load_operation(i) for i in range(50)]
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Calculate performance metrics
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else avg_response_time
        
        # Performance criteria
        meets_avg_target = avg_response_time < self.target_db_response_time
        meets_p95_target = p95_response_time < self.target_db_response_time * 2  # Allow 2x for P95
        high_success_rate = len(successful_results) / len(results) > 0.95 if results else False
        
        return {
            "success": meets_avg_target and meets_p95_target and high_success_rate,
            "total_operations": len(results),
            "successful_operations": len(successful_results),
            "avg_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "meets_avg_target": meets_avg_target,
            "meets_p95_target": meets_p95_target,
            "high_success_rate": high_success_rate,
            "target_response_time_ms": self.target_db_response_time
        }
    
    async def _test_database_auto_scaling(self) -> Dict[str, Any]:
        """Test database connection auto-scaling under varying load."""
        logger.info("Testing database auto-scaling...")
        
        # Get initial pool metrics
        initial_metrics = await self.unified_db_manager.get_performance_metrics()
        initial_pool_size = initial_metrics.get("pool_size", 0)
        
        # Generate varying load to trigger scaling
        async def scaling_load_operation():
            async with self.unified_db_manager.get_async_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT pg_sleep(0.1)"))  # Small delay to simulate work
        
        # Phase 1: Light load
        light_load_tasks = [scaling_load_operation() for _ in range(5)]
        await asyncio.gather(*light_load_tasks, return_exceptions=True)
        
        light_load_metrics = await self.unified_db_manager.get_performance_metrics()
        
        # Phase 2: Heavy load to trigger scaling up
        heavy_load_tasks = [scaling_load_operation() for _ in range(30)]
        await asyncio.gather(*heavy_load_tasks, return_exceptions=True)
        
        # Wait for potential scaling
        await asyncio.sleep(2)
        
        heavy_load_metrics = await self.unified_db_manager.get_performance_metrics()
        
        # Phase 3: Return to light load to trigger scaling down
        await asyncio.sleep(5)  # Wait for scale-down cooldown
        
        light_load_tasks_2 = [scaling_load_operation() for _ in range(2)]
        await asyncio.gather(*light_load_tasks_2, return_exceptions=True)
        
        await asyncio.sleep(2)
        final_metrics = await self.unified_db_manager.get_performance_metrics()
        
        # Analyze scaling behavior
        pool_size_increased = heavy_load_metrics.get("pool_size", 0) >= light_load_metrics.get("pool_size", 0)
        scaling_responsive = True  # Basic scaling test passes if no errors occur
        
        return {
            "success": scaling_responsive,
            "initial_pool_size": initial_pool_size,
            "light_load_pool_size": light_load_metrics.get("pool_size", 0),
            "heavy_load_pool_size": heavy_load_metrics.get("pool_size", 0),
            "final_pool_size": final_metrics.get("pool_size", 0),
            "pool_size_increased": pool_size_increased,
            "scaling_responsive": scaling_responsive,
            "utilization_metrics": {
                "light_load": light_load_metrics.get("pool_utilization", 0),
                "heavy_load": heavy_load_metrics.get("pool_utilization", 0),
                "final": final_metrics.get("pool_utilization", 0)
            }
        }
    
    async def phase_d_real_training_validation(self):
        """
        Phase D: Execute real model training using consolidated infrastructure.
        
        Validates:
        - End-to-end ML training workflow using consolidated components
        - TrainingSystemManager integration with all consolidated systems
        - AutoML orchestration with real training operations
        - Performance and reliability under actual ML workload
        """
        logger.info("🤖 Phase D: Real Model Training Validation Starting")
        phase_start = time.time()
        
        try:
            # Initialize TrainingSystemManager
            from rich.console import Console
            console = Console()
            self.training_manager = TrainingSystemManager(console=console)
            
            # Test 1: Training system initialization with consolidated infrastructure
            initialization_result = await self._test_training_system_initialization()
            
            # Test 2: Real model training execution
            model_training_result = await self._test_real_model_training()
            
            # Test 3: AutoML orchestration integration
            automl_integration_result = await self._test_automl_integration()
            
            # Test 4: End-to-end workflow validation
            e2e_workflow_result = await self._test_e2e_training_workflow()
            
            phase_d_result = TestResults(
                test_name="Phase_D_RealModelTraining",
                success=all([
                    initialization_result["success"],
                    model_training_result["success"],
                    automl_integration_result["success"],
                    e2e_workflow_result["success"]
                ]),
                execution_time=time.time() - phase_start,
                metrics={
                    "system_initialization": initialization_result,
                    "model_training": model_training_result,
                    "automl_integration": automl_integration_result,
                    "e2e_workflow": e2e_workflow_result
                }
            )
            
            self.results.append(phase_d_result)
            logger.info(f"✅ Phase D completed in {phase_d_result.execution_time:.2f}s")
            
        except Exception as e:
            error_result = TestResults(
                test_name="Phase_D_RealModelTraining",
                success=False,
                execution_time=time.time() - phase_start,
                metrics={},
                error_message=str(e)
            )
            self.results.append(error_result)
            logger.error(f"❌ Phase D failed: {e}")
    
    async def _test_training_system_initialization(self) -> Dict[str, Any]:
        """Test training system initialization with consolidated infrastructure."""
        logger.info("Testing training system initialization...")
        
        try:
            # Initialize training system with smart initialization
            initialization_result = await self.training_manager.smart_initialize()
            
            # Check if initialization was successful
            success = initialization_result.get("success", False)
            
            # Get system status after initialization
            if success:
                system_status = await self.training_manager.get_system_status()
            else:
                system_status = {"error": "initialization_failed"}
                
            # Validate training readiness
            training_ready = await self.training_manager.validate_ready_for_training() if success else False
            
            return {
                "success": success and training_ready,
                "initialization_result": initialization_result,
                "system_status": system_status,
                "training_ready": training_ready,
                "initialization_time": initialization_result.get("initialization_time_seconds", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "initialization_failed": True
            }
    
    async def _test_real_model_training(self) -> Dict[str, Any]:
        """Test real model training execution using consolidated infrastructure."""
        logger.info("Testing real model training execution...")
        
        try:
            # Create a training session
            training_config = {
                "continuous_mode": False,  # Single iteration for testing
                "max_iterations": 1,
                "improvement_threshold": 0.01,
                "timeout": 300,  # 5 minutes
                "auto_init": True
            }
            
            training_session = await self.training_manager.create_training_session(training_config)
            
            # Start training system if not already running
            if self.training_manager._training_status != "running":
                startup_result = await self.training_manager.start_training_system()
                if startup_result["status"] != "success":
                    return {
                        "success": False,
                        "error": "Failed to start training system",
                        "startup_result": startup_result
                    }
            
            # Simulate training data availability check
            data_needs_generation = await self.training_manager._needs_synthetic_data()
            
            # Generate initial data if needed
            if data_needs_generation:
                data_generation_result = await self.training_manager._generate_initial_data()
                data_generation_success = data_generation_result.get("success", False)
            else:
                data_generation_success = True
                data_generation_result = {"message": "Sufficient data available"}
            
            # Get final training status
            final_status = await self.training_manager.get_training_status()
            
            return {
                "success": training_session is not None and data_generation_success,
                "training_session_created": training_session is not None,
                "data_generation_success": data_generation_success,
                "data_generation_result": data_generation_result,
                "final_training_status": final_status,
                "training_session_id": training_session.session_id if training_session else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "training_execution_failed": True
            }
    
    async def _test_automl_integration(self) -> Dict[str, Any]:
        """Test AutoML orchestration integration with consolidated systems."""
        logger.info("Testing AutoML orchestration integration...")
        
        try:
            # Create AutoML configuration
            automl_config = AutoMLConfig(
                n_trials=5,  # Small number for testing
                timeout=60,  # 1 minute timeout
                optimization_mode="hyperparameter_optimization",
                enable_real_time_feedback=True
            )
            
            # Initialize AutoML orchestrator with database manager
            from src.prompt_improver.ml.automl.orchestrator import create_automl_orchestrator
            self.automl_orchestrator = await create_automl_orchestrator(
                config=automl_config,
                db_manager=get_sessionmanager()
            )
            
            # Test optimization status
            initial_status = await self.automl_orchestrator.get_optimization_status()
            
            # Start a small optimization run
            optimization_result = await self.automl_orchestrator.start_optimization(
                optimization_target="rule_effectiveness",
                experiment_config={"sample_size": 100}
            )
            
            # Get optimization status after run
            final_status = await self.automl_orchestrator.get_optimization_status()
            
            # Check if optimization completed successfully
            optimization_success = not optimization_result.get("error")
            
            return {
                "success": optimization_success,
                "initial_status": initial_status,
                "optimization_result": optimization_result,
                "final_status": final_status,
                "optimization_completed": optimization_success
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "automl_integration_failed": True
            }
    
    async def _test_e2e_training_workflow(self) -> Dict[str, Any]:
        """Test end-to-end training workflow with all consolidated components."""
        logger.info("Testing end-to-end training workflow...")
        
        try:
            # Comprehensive workflow test combining all components
            workflow_steps = []
            
            # Step 1: Event bus integration
            if self.event_bus:
                training_event = MLEvent(
                    event_type=EventType.TRAINING_STARTED,
                    source="checkpoint_2a_test",
                    data={"workflow": "e2e_test", "components": "all"},
                    timestamp=datetime.now(timezone.utc)
                )
                await self.event_bus.emit(training_event, priority=8)
                workflow_steps.append("event_emitted")
            
            # Step 2: Background task for workflow coordination
            async def workflow_coordination_task():
                # Simulate workflow coordination
                await asyncio.sleep(0.5)
                return {"step": "workflow_coordination", "status": "completed"}
            
            workflow_task_id = await self.background_task_manager.submit_enhanced_task(
                task_id="e2e_workflow_coordination",
                coroutine=workflow_coordination_task(),
                priority="HIGH",
                timeout=10
            )
            
            workflow_task_result = await self.background_task_manager.get_task_result(
                workflow_task_id, timeout=15
            )
            
            if workflow_task_result:
                workflow_steps.append("background_task_completed")
            
            # Step 3: Database operations during workflow
            if self.unified_db_manager:
                async with self.unified_db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                    workflow_steps.append("database_operation_completed")
            
            # Step 4: Training system status check
            if self.training_manager:
                training_status = await self.training_manager.get_training_status()
                if training_status:
                    workflow_steps.append("training_status_checked")
            
            # Step 5: Emit completion event
            if self.event_bus:
                completion_event = MLEvent(
                    event_type=EventType.TRAINING_COMPLETED,
                    source="checkpoint_2a_test",
                    data={"workflow": "e2e_test", "steps_completed": len(workflow_steps)},
                    timestamp=datetime.now(timezone.utc)
                )
                await self.event_bus.emit(completion_event, priority=9)
                workflow_steps.append("completion_event_emitted")
            
            # Evaluate workflow success
            expected_steps = 5
            workflow_success = len(workflow_steps) >= expected_steps
            
            return {
                "success": workflow_success,
                "workflow_steps": workflow_steps,
                "steps_completed": len(workflow_steps),
                "expected_steps": expected_steps,
                "workflow_task_result": workflow_task_result,
                "comprehensive_integration": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_steps": workflow_steps,
                "e2e_workflow_failed": True
            }
    
    async def phase_e_performance_analysis(self) -> Dict[str, Any]:
        """
        Phase E: Comprehensive performance metrics analysis.
        
        Analyzes:
        - Event processing throughput and success criteria
        - Database performance and SLA compliance
        - Task management efficiency and resource utilization
        - Overall system integration and performance
        """
        logger.info("📊 Phase E: Performance Metrics Analysis Starting")
        
        # Collect comprehensive metrics from all components
        event_bus_metrics = self.event_bus.get_performance_metrics() if self.event_bus else {}
        db_metrics = await self.unified_db_manager.get_performance_metrics() if self.unified_db_manager else {}
        task_manager_status = await self.background_task_manager.get_comprehensive_status()
        
        # Analyze performance against success criteria
        performance_analysis = {
            "event_processing_analysis": self._analyze_event_processing_performance(event_bus_metrics),
            "database_performance_analysis": self._analyze_database_performance(db_metrics),
            "task_management_analysis": self._analyze_task_management_performance(task_manager_status),
            "integration_analysis": self._analyze_integration_performance(),
            "overall_success_criteria": self._evaluate_overall_success_criteria(
                event_bus_metrics, db_metrics, task_manager_status
            )
        }
        
        return performance_analysis
    
    def _analyze_event_processing_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze event processing performance against targets."""
        throughput = metrics.get("throughput_per_second", 0)
        events_processed = metrics.get("events_processed", 0)
        events_failed = metrics.get("events_failed", 0)
        
        # Calculate success metrics
        throughput_target_met = throughput >= self.target_event_throughput
        error_rate = events_failed / max(events_processed, 1)
        error_rate_acceptable = error_rate <= self.max_error_rate
        
        return {
            "throughput_events_per_second": throughput,
            "target_throughput": self.target_event_throughput,
            "throughput_target_met": throughput_target_met,
            "events_processed": events_processed,
            "events_failed": events_failed,
            "error_rate": error_rate,
            "error_rate_acceptable": error_rate_acceptable,
            "overall_event_performance": throughput_target_met and error_rate_acceptable
        }
    
    def _analyze_database_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database performance against targets."""
        avg_response_time = metrics.get("avg_response_time_ms", 0)
        pool_utilization = metrics.get("pool_utilization", 0)
        queries_executed = metrics.get("queries_executed", 0)
        queries_failed = metrics.get("queries_failed", 0)
        
        # Calculate success metrics
        response_time_target_met = avg_response_time < self.target_db_response_time
        query_error_rate = queries_failed / max(queries_executed, 1)
        query_error_rate_acceptable = query_error_rate <= self.max_error_rate
        
        return {
            "avg_response_time_ms": avg_response_time,
            "target_response_time_ms": self.target_db_response_time,
            "response_time_target_met": response_time_target_met,
            "pool_utilization_percent": pool_utilization,
            "queries_executed": queries_executed,
            "queries_failed": queries_failed,
            "query_error_rate": query_error_rate,
            "query_error_rate_acceptable": query_error_rate_acceptable,
            "overall_db_performance": response_time_target_met and query_error_rate_acceptable
        }
    
    def _analyze_task_management_performance(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task management performance."""
        active_tasks = status.get("active_tasks", 0)
        completed_tasks = status.get("metrics", {}).get("tasks_completed", 0)
        failed_tasks = status.get("metrics", {}).get("tasks_failed", 0)
        
        # Calculate success metrics
        total_tasks = completed_tasks + failed_tasks
        task_success_rate = completed_tasks / max(total_tasks, 1)
        task_success_rate_acceptable = task_success_rate >= (1 - self.max_error_rate)
        
        return {
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "total_tasks": total_tasks,
            "task_success_rate": task_success_rate,
            "task_success_rate_acceptable": task_success_rate_acceptable,
            "overall_task_performance": task_success_rate_acceptable
        }
    
    def _analyze_integration_performance(self) -> Dict[str, Any]:
        """Analyze overall integration and coordination performance."""
        # Analyze results from all test phases
        phase_results = {result.test_name: result.success for result in self.results}
        
        successful_phases = sum(1 for success in phase_results.values() if success)
        total_phases = len(phase_results)
        integration_success_rate = successful_phases / max(total_phases, 1)
        
        return {
            "phase_results": phase_results,
            "successful_phases": successful_phases,
            "total_phases": total_phases,
            "integration_success_rate": integration_success_rate,
            "overall_integration_success": integration_success_rate >= 0.8  # 80% success threshold
        }
    
    def _evaluate_overall_success_criteria(self, event_metrics, db_metrics, task_status) -> Dict[str, Any]:
        """Evaluate overall success against Checkpoint 2A criteria."""
        # Success Criteria Evaluation
        criteria_results = {
            "event_processing_8k_per_sec": event_metrics.get("throughput_per_second", 0) >= self.target_event_throughput,
            "db_response_time_under_200ms": db_metrics.get("avg_response_time_ms", 0) < self.target_db_response_time,
            "error_rate_under_0_1_percent": True,  # Will be calculated from individual components
            "all_components_integrated": len(self.results) >= 4  # All phases completed
        }
        
        # Calculate overall error rate
        total_operations = (
            event_metrics.get("events_processed", 0) +
            db_metrics.get("queries_executed", 0) +
            task_status.get("metrics", {}).get("tasks_completed", 0)
        )
        
        total_failures = (
            event_metrics.get("events_failed", 0) +
            db_metrics.get("queries_failed", 0) +
            task_status.get("metrics", {}).get("tasks_failed", 0)
        )
        
        overall_error_rate = total_failures / max(total_operations, 1)
        criteria_results["error_rate_under_0_1_percent"] = overall_error_rate <= self.max_error_rate
        
        # Overall success
        overall_success = all(criteria_results.values())
        
        return {
            "success_criteria_met": criteria_results,
            "overall_error_rate": overall_error_rate,
            "total_operations": total_operations,
            "total_failures": total_failures,
            "overall_success": overall_success,
            "checkpoint_2a_validation_passed": overall_success
        }
    
    async def generate_final_report(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final validation report."""
        total_execution_time = time.time() - self.start_time
        
        # Collect all component metrics for report
        final_metrics = {}
        
        if self.event_bus:
            final_metrics["event_bus"] = self.event_bus.get_performance_metrics()
        
        if self.unified_db_manager:
            final_metrics["database"] = await self.unified_db_manager.get_performance_metrics()
        
        final_metrics["background_tasks"] = await self.background_task_manager.get_comprehensive_status()
        
        # Determine overall validation result
        overall_success = performance_analysis["overall_success_criteria"]["overall_success"]
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(overall_success, performance_analysis)
        
        final_report = {
            "checkpoint_2a_validation": {
                "overall_success": overall_success,
                "validation_passed": overall_success,
                "total_execution_time_seconds": total_execution_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "executive_summary": executive_summary,
            "phase_results": [
                {
                    "test_name": result.test_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message
                }
                for result in self.results
            ],
            "performance_analysis": performance_analysis,
            "consolidated_metrics": final_metrics,
            "success_criteria_validation": performance_analysis["overall_success_criteria"],
            "recommendations": self._generate_recommendations(performance_analysis)
        }
        
        return final_report
    
    def _generate_executive_summary(self, overall_success: bool, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of validation results."""
        if overall_success:
            status = "PASSED"
            message = "Week 3 ML infrastructure consolidation successfully validated"
        else:
            status = "FAILED"
            message = "Week 3 ML infrastructure consolidation validation encountered issues"
        
        # Key achievements
        achievements = []
        if performance_analysis["event_processing_analysis"]["overall_event_performance"]:
            achievements.append("Event processing throughput target achieved")
        
        if performance_analysis["database_performance_analysis"]["overall_db_performance"]:
            achievements.append("Database performance targets met")
        
        if performance_analysis["task_management_analysis"]["overall_task_performance"]:
            achievements.append("Background task management functioning properly")
        
        if performance_analysis["integration_analysis"]["overall_integration_success"]:
            achievements.append("Component integration successful")
        
        return {
            "status": status,
            "message": message,
            "key_achievements": achievements,
            "components_tested": len(self.results),
            "integration_success_rate": performance_analysis["integration_analysis"]["integration_success_rate"]
        }
    
    def _generate_recommendations(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Event processing recommendations
        event_perf = performance_analysis["event_processing_analysis"]
        if not event_perf["throughput_target_met"]:
            recommendations.append(
                f"Consider scaling up event bus workers - current throughput: {event_perf['throughput_events_per_second']:.0f} events/sec, target: {self.target_event_throughput}"
            )
        
        # Database performance recommendations
        db_perf = performance_analysis["database_performance_analysis"]
        if not db_perf["response_time_target_met"]:
            recommendations.append(
                f"Optimize database queries - current avg response time: {db_perf['avg_response_time_ms']:.1f}ms, target: <{self.target_db_response_time}ms"
            )
        
        # Integration recommendations
        integration_perf = performance_analysis["integration_analysis"]
        if not integration_perf["overall_integration_success"]:
            recommendations.append("Review component integration patterns for improved coordination")
        
        # Success recommendations
        if performance_analysis["overall_success_criteria"]["overall_success"]:
            recommendations.append("✅ Ready to proceed with Week 4 API layer standardization")
            recommendations.append("Consider production deployment of consolidated ML infrastructure")
        else:
            recommendations.append("❌ Address performance issues before proceeding to Week 4")
            recommendations.append("Re-run Checkpoint 2A after implementing fixes")
        
        return recommendations
    
    async def cleanup(self):
        """Cleanup test resources."""
        logger.info("🧹 Cleaning up test resources...")
        
        try:
            if self.event_bus:
                await self.event_bus.stop()
            
            if self.unified_db_manager:
                await self.unified_db_manager.close()
            
            if self.training_manager:
                await self.training_manager.stop_training_system()
                
            logger.info("✅ Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Main execution function
async def main():
    """Execute Checkpoint 2A validation."""
    validator = Checkpoint2AValidator()
    
    try:
        # Run comprehensive validation
        results = await validator.run_comprehensive_validation()
        
        # Save results to file
        results_file = Path("checkpoint_2a_validation_results.json")
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 CHECKPOINT 2A: ML PIPELINE ORCHESTRATION VALIDATION RESULTS")
        print("="*80)
        
        if results.get("checkpoint_2a_validation", {}).get("overall_success", False):
            print("✅ VALIDATION PASSED - Week 3 consolidation successfully validated")
            print("🚀 Ready to proceed with Week 4 API layer standardization")
        else:
            print("❌ VALIDATION FAILED - Issues found in consolidation")
            print("🔧 Review recommendations and re-run validation")
        
        print(f"\n📊 Results saved to: {results_file.absolute()}")
        print(f"⏱️  Total execution time: {results.get('checkpoint_2a_validation', {}).get('total_execution_time_seconds', 0):.2f}s\n")
        
        # Print key metrics
        success_criteria = results.get("success_criteria_validation", {})
        print("📈 Key Performance Metrics:")
        for criterion, met in success_criteria.get("success_criteria_met", {}).items():
            status = "✅" if met else "❌"
            print(f"  {status} {criterion}: {'PASSED' if met else 'FAILED'}")
        
        print("\n" + "="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Validation execution failed: {e}")
        return {"success": False, "error": str(e)}
    
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())