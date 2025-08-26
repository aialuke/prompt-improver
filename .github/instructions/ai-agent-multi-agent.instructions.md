# AI Agent and Multi-Agent System Instructions for APES

## Overview
This file provides GitHub Copilot with specific guidance for building AI agents and multi-agent systems within the Adaptive Prompt Enhancement System (APES). Based on 2025 best practices including AutoGen patterns, MCP integrations, and agent orchestration frameworks.

## Multi-Agent Architecture Patterns

### 1. Agent Composition Framework
Use the actor-critic pattern for complex reasoning tasks:

```python
# Good: Structured multi-agent system
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

class AgentRole(Enum):
    ANALYZER = "analyzer"
    CRITIC = "critic"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"

@dataclass
class AgentMessage:
    id: str
    sender: str
    recipient: str
    content: str
    message_type: str
    metadata: Dict[str, Any]
    timestamp: float

class BaseAgent(ABC):
    def __init__(self, name: str, role: AgentRole, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.message_history: List[AgentMessage] = []
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        pass
    
    @abstractmethod
    async def can_handle(self, task: str) -> bool:
        pass

class PromptAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="prompt_analyzer",
            role=AgentRole.ANALYZER,
            capabilities=["prompt_evaluation", "performance_analysis", "optimization"]
        )
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type == "analyze_prompt":
            analysis = await self._analyze_prompt(message.content)
            return AgentMessage(
                id=f"analysis_{uuid.uuid4()}",
                sender=self.name,
                recipient=message.sender,
                content=analysis,
                message_type="analysis_result",
                metadata={"confidence": 0.95},
                timestamp=time.time()
            )
    
    async def _analyze_prompt(self, prompt: str) -> str:
        # Implement prompt analysis logic
        return f"Analysis of prompt: {prompt[:100]}..."
```

### 2. Agent Orchestration with MCP
Integrate agents with Model Context Protocol servers:

```python
# Good: MCP-integrated agent system
import asyncio
from mcp import ClientSession, StdioServerParameters
from typing import AsyncGenerator

class MCPIntegratedAgent(BaseAgent):
    def __init__(self, name: str, role: AgentRole, mcp_servers: Dict[str, str]):
        super().__init__(name, role, [])
        self.mcp_servers = mcp_servers
        self.mcp_sessions: Dict[str, ClientSession] = {}
    
    async def initialize_mcp_connections(self):
        """Initialize connections to MCP servers"""
        for server_name, server_path in self.mcp_servers.items():
            try:
                session = ClientSession(StdioServerParameters(
                    command=server_path,
                    args=[]
                ))
                await session.initialize()
                self.mcp_sessions[server_name] = session
                
                # Get server capabilities
                capabilities = await session.get_capabilities()
                self.capabilities.extend(capabilities.get("tools", []))
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_name}: {e}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool through MCP"""
        for session in self.mcp_sessions.values():
            try:
                result = await session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed on session: {e}")
        
        raise RuntimeError(f"No MCP session could execute tool: {tool_name}")

class PromptOptimizationOrchestrator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, Any] = {}
    
    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
    
    async def orchestrate_optimization(self, prompt: str, optimization_goals: List[str]) -> Dict[str, Any]:
        """Orchestrate multi-agent prompt optimization"""
        task_id = f"opt_{uuid.uuid4()}"
        
        # Phase 1: Analysis
        analysis_tasks = [
            self._analyze_prompt_structure(prompt),
            self._analyze_prompt_performance(prompt),
            self._analyze_prompt_clarity(prompt)
        ]
        
        analyses = await asyncio.gather(*analysis_tasks)
        
        # Phase 2: Optimization Generation
        optimization_agents = [
            agent for agent in self.agents.values()
            if agent.role == AgentRole.SPECIALIST
        ]
        
        optimization_tasks = [
            agent.process_message(AgentMessage(
                id=f"opt_req_{uuid.uuid4()}",
                sender="orchestrator",
                recipient=agent.name,
                content=prompt,
                message_type="optimization_request",
                metadata={"goals": optimization_goals, "analyses": analyses},
                timestamp=time.time()
            ))
            for agent in optimization_agents
        ]
        
        optimizations = await asyncio.gather(*optimization_tasks)
        
        # Phase 3: Critique and Selection
        critic_agent = next(
            (agent for agent in self.agents.values() if agent.role == AgentRole.CRITIC),
            None
        )
        
        if critic_agent:
            best_optimization = await critic_agent.process_message(AgentMessage(
                id=f"critique_{uuid.uuid4()}",
                sender="orchestrator",
                recipient=critic_agent.name,
                content=json.dumps([opt.content for opt in optimizations]),
                message_type="critique_request",
                metadata={"original": prompt, "criteria": optimization_goals},
                timestamp=time.time()
            ))
            
            return {
                "task_id": task_id,
                "original_prompt": prompt,
                "optimized_prompt": best_optimization.content,
                "analyses": [analysis.content for analysis in analyses],
                "alternatives": [opt.content for opt in optimizations],
                "confidence": best_optimization.metadata.get("confidence", 0.0)
            }
        
        return {
            "task_id": task_id,
            "error": "No critic agent available for final selection"
        }
```

### 3. Conversational Agent Patterns
Implement context-aware conversational agents:

```python
# Good: Context-aware conversational agent
class ConversationalAgent(BaseAgent):
    def __init__(self, name: str, persona: str, context_window: int = 10):
        super().__init__(name, AgentRole.SPECIALIST, ["conversation", "context_management"])
        self.persona = persona
        self.context_window = context_window
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_embeddings: Dict[str, List[float]] = {}
    
    async def process_conversation(self, user_input: str, context: Dict[str, Any]) -> str:
        """Process conversational input with context awareness"""
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": time.time(),
            "user_input": user_input,
            "context": context,
            "session_id": context.get("session_id")
        })
        
        # Maintain context window
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
        
        # Generate contextual response
        relevant_context = await self._get_relevant_context(user_input)
        response = await self._generate_response(user_input, relevant_context)
        
        # Update context embeddings for future retrieval
        await self._update_context_embeddings(user_input, response)
        
        return response
    
    async def _get_relevant_context(self, input_text: str) -> Dict[str, Any]:
        """Retrieve relevant context based on semantic similarity"""
        # Implement semantic search through conversation history
        # This would typically use vector embeddings and similarity search
        return {
            "recent_context": self.conversation_history[-3:],
            "relevant_topics": await self._extract_topics(input_text),
            "user_preferences": await self._get_user_preferences()
        }
    
    async def _generate_response(self, input_text: str, context: Dict[str, Any]) -> str:
        """Generate contextually appropriate response"""
        system_prompt = f"""
        You are {self.persona}. 
        
        Conversation context: {json.dumps(context["recent_context"], indent=2)}
        Relevant topics: {context["relevant_topics"]}
        User preferences: {context["user_preferences"]}
        
        Respond to: {input_text}
        
        Guidelines:
        - Maintain consistency with previous conversation
        - Reference relevant context when appropriate
        - Stay in character as {self.persona}
        - Provide helpful, actionable responses
        """
        
        # This would call your LLM of choice
        response = await self._call_llm(system_prompt)
        return response
```

### 4. Task-Specific Agent Specialization
Create specialized agents for different APES tasks:

```python
# Good: Specialized agents for APES tasks
class PromptValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="prompt_validator",
            role=AgentRole.SPECIALIST,
            capabilities=["syntax_validation", "semantic_analysis", "safety_check"]
        )
        self.validation_rules = self._load_validation_rules()
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type == "validate_prompt":
            validation_result = await self._validate_prompt(message.content)
            return AgentMessage(
                id=f"validation_{uuid.uuid4()}",
                sender=self.name,
                recipient=message.sender,
                content=json.dumps(validation_result),
                message_type="validation_result",
                metadata={"validation_passed": validation_result["is_valid"]},
                timestamp=time.time()
            )
    
    async def _validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Comprehensive prompt validation"""
        results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "safety_score": 1.0,
            "clarity_score": 1.0
        }
        
        # Syntax validation
        syntax_errors = await self._check_syntax(prompt)
        if syntax_errors:
            results["errors"].extend(syntax_errors)
            results["is_valid"] = False
        
        # Semantic analysis
        semantic_issues = await self._analyze_semantics(prompt)
        results["warnings"].extend(semantic_issues)
        
        # Safety check
        safety_score = await self._safety_check(prompt)
        results["safety_score"] = safety_score
        if safety_score < 0.8:
            results["errors"].append("Prompt may contain unsafe content")
            results["is_valid"] = False
        
        # Clarity assessment
        clarity_score = await self._assess_clarity(prompt)
        results["clarity_score"] = clarity_score
        if clarity_score < 0.6:
            results["warnings"].append("Prompt clarity could be improved")
        
        # Generate suggestions
        if not results["is_valid"] or results["warnings"]:
            results["suggestions"] = await self._generate_suggestions(prompt, results)
        
        return results

class PerformanceOptimizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="performance_optimizer",
            role=AgentRole.SPECIALIST,
            capabilities=["token_optimization", "latency_reduction", "cost_optimization"]
        )
        self.performance_metrics = {}
    
    async def optimize_for_performance(self, prompt: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize prompt for performance metrics"""
        
        # Analyze current performance
        current_metrics = await self._analyze_current_performance(prompt)
        
        # Apply optimization strategies
        optimizations = []
        
        if constraints.get("optimize_tokens", True):
            token_optimized = await self._optimize_tokens(prompt)
            optimizations.append({
                "strategy": "token_optimization",
                "optimized_prompt": token_optimized,
                "token_reduction": current_metrics["tokens"] - len(token_optimized.split())
            })
        
        if constraints.get("optimize_latency", True):
            latency_optimized = await self._optimize_latency(prompt)
            optimizations.append({
                "strategy": "latency_optimization", 
                "optimized_prompt": latency_optimized,
                "estimated_speedup": "15-25%"
            })
        
        if constraints.get("optimize_cost", True):
            cost_optimized = await self._optimize_cost(prompt)
            optimizations.append({
                "strategy": "cost_optimization",
                "optimized_prompt": cost_optimized,
                "cost_reduction": "10-20%"
            })
        
        # Select best optimization based on constraints
        best_optimization = await self._select_best_optimization(optimizations, constraints)
        
        return {
            "original_prompt": prompt,
            "optimized_prompt": best_optimization["optimized_prompt"],
            "optimization_strategy": best_optimization["strategy"],
            "performance_improvement": best_optimization,
            "all_optimizations": optimizations
        }
```

### 5. Agent Communication Protocols
Implement robust communication between agents:

```python
# Good: Structured agent communication
class AgentCommunicationManager:
    def __init__(self):
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
    
    def register_agent(self, agent: BaseAgent):
        self.agent_registry[agent.name] = agent
        
    def register_message_handler(self, message_type: str, handler: Callable):
        self.message_handlers[message_type] = handler
    
    async def start(self):
        """Start the communication manager"""
        self.running = True
        await asyncio.create_task(self._message_processor())
    
    async def send_message(self, message: AgentMessage):
        """Send message through the communication bus"""
        await self.message_bus.put(message)
    
    async def broadcast_message(self, message: AgentMessage, exclude: List[str] = None):
        """Broadcast message to all agents except excluded ones"""
        exclude = exclude or []
        for agent_name in self.agent_registry:
            if agent_name not in exclude:
                message_copy = AgentMessage(
                    id=f"{message.id}_broadcast_{agent_name}",
                    sender=message.sender,
                    recipient=agent_name,
                    content=message.content,
                    message_type=message.message_type,
                    metadata=message.metadata.copy(),
                    timestamp=time.time()
                )
                await self.send_message(message_copy)
    
    async def _message_processor(self):
        """Process messages from the communication bus"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_bus.get(), timeout=1.0)
                await self._route_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _route_message(self, message: AgentMessage):
        """Route message to appropriate recipient"""
        if message.recipient in self.agent_registry:
            agent = self.agent_registry[message.recipient]
            try:
                response = await agent.process_message(message)
                if response and response.recipient != "none":
                    await self.send_message(response)
            except Exception as e:
                logger.error(f"Agent {message.recipient} failed to process message: {e}")
        
        # Handle special message types
        if message.message_type in self.message_handlers:
            try:
                # Ensure the handler exists and is callable before invoking
                handler = self.message_handlers.get(message.message_type)
                if handler is not None and callable(handler):
                    await handler(message)
                else:
                    logger.error(f"No handler found for message type: {message.message_type}")
            except Exception as e:
                logger.error(f"Message handler for {message.message_type} failed: {e}")
```

## AutoGen Integration Patterns

### 1. AutoGen-Style Group Chat
```python
# Good: AutoGen-style agent coordination
class APESGroupChat:
    def __init__(self, agents: List[BaseAgent], admin_agent: Optional[BaseAgent] = None):
        self.agents = {agent.name: agent for agent in agents}
        self.admin_agent = admin_agent
        self.conversation_history: List[AgentMessage] = []
        self.max_rounds = 10
    
    async def initiate_chat(self, initial_message: str, task_description: str) -> List[AgentMessage]:
        """Initiate group chat for collaborative problem solving"""
        
        # Start with initial message
        current_message = AgentMessage(
            id=f"init_{uuid.uuid4()}",
            sender="user",
            recipient="all",
            content=initial_message,
            message_type="task_initiation",
            metadata={"task": task_description},
            timestamp=time.time()
        )
        
        self.conversation_history.append(current_message)
        
        for round_num in range(self.max_rounds):
            # Select next speaker
            next_speaker = await self._select_next_speaker(current_message)
            if not next_speaker:
                break
            
            # Generate response
            response = await next_speaker.process_message(current_message)
            self.conversation_history.append(response)
            current_message = response
            
            # Check if task is complete
            if await self._is_task_complete(response):
                break
        
        return self.conversation_history
    
    async def _select_next_speaker(self, last_message: AgentMessage) -> Optional[BaseAgent]:
        """Select next agent to speak based on context and capabilities"""
        if self.admin_agent:
            selection_result = await self.admin_agent.process_message(AgentMessage(
                id=f"select_{uuid.uuid4()}",
                sender="system",
                recipient=self.admin_agent.name,
                content=json.dumps({
                    "last_message": last_message.content,
                    "available_agents": list(self.agents.keys()),
                    "conversation_context": [msg.content for msg in self.conversation_history[-3:]]
                }),
                message_type="speaker_selection",
                metadata={},
                timestamp=time.time()
            ))
            
            selected_name = selection_result.content
            return self.agents.get(selected_name)
        
        # Simple round-robin if no admin agent
        last_speaker_idx = list(self.agents.keys()).index(last_message.sender)
        next_idx = (last_speaker_idx + 1) % len(self.agents)
        return list(self.agents.values())[next_idx]
```

### 2. Human-in-the-Loop Integration
```python
# Good: Human-in-the-loop agent system
class HumanInTheLoopAgent(BaseAgent):
    def __init__(self, interface_type: str = "web"):
        super().__init__(
            name="human_reviewer",
            role=AgentRole.CRITIC,
            capabilities=["human_feedback", "decision_making", "quality_assessment"]
        )
        self.interface_type = interface_type
        self.pending_reviews: Dict[str, Any] = {}
    
    async def request_human_review(self, item_to_review: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Request human review with structured interface"""
        review_id = f"review_{uuid.uuid4()}"
        
        review_request = {
            "id": review_id,
            "item": item_to_review,
            "context": context,
            "timestamp": time.time(),
            "status": "pending"
        }
        
        self.pending_reviews[review_id] = review_request
        
        # Send to appropriate interface
        if self.interface_type == "web":
            await self._send_to_web_interface(review_request)
        elif self.interface_type == "slack":
            await self._send_to_slack(review_request)
        elif self.interface_type == "email":
            await self._send_to_email(review_request)
        
        # Wait for human response (with timeout)
        try:
            response = await asyncio.wait_for(
                self._wait_for_human_response(review_id),
                timeout=3600  # 1 hour timeout
            )
            return response
        except asyncio.TimeoutError:
            return {
                "review_id": review_id,
                "status": "timeout",
                "default_action": "approve",  # or whatever default makes sense
                "message": "Human review timed out, proceeding with default action"
            }
    
    async def _send_to_web_interface(self, review_request: Dict[str, Any]):
        """Send review request to web interface"""
        # Implementation would depend on your web framework
        # This might push to a queue, update a database, send a websocket message, etc.
        pass
    
    async def receive_human_feedback(self, review_id: str, feedback: Dict[str, Any]):
        """Receive feedback from human reviewer"""
        if review_id in self.pending_reviews:
            self.pending_reviews[review_id]["feedback"] = feedback
            self.pending_reviews[review_id]["status"] = "completed"
            self.pending_reviews[review_id]["completed_at"] = time.time()
```

## Best Practices for Multi-Agent Systems

### 1. Error Handling and Resilience
```python
# Good: Robust error handling in agent systems
class ResilientAgent(BaseAgent):
    def __init__(self, name: str, role: AgentRole, max_retries: int = 3):
        super().__init__(name, role, [])
        self.max_retries = max_retries
        self.error_counts: Dict[str, int] = {}
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300  # 5 minutes
        self.circuit_breakers: Dict[str, float] = {}
    
    async def process_message_with_retry(self, message: AgentMessage) -> AgentMessage:
        """Process message with retry logic and circuit breaker"""
        operation_key = f"{message.message_type}_{message.sender}"
        
        # Check circuit breaker
        if await self._is_circuit_open(operation_key):
            return AgentMessage(
                id=f"error_{uuid.uuid4()}",
                sender=self.name,
                recipient=message.sender,
                content="Service temporarily unavailable (circuit breaker open)",
                message_type="error",
                metadata={"error_type": "circuit_breaker"},
                timestamp=time.time()
            )
        
        for attempt in range(self.max_retries):
            try:
                result = await self.process_message(message)
                # Reset error count on success
                self.error_counts[operation_key] = 0
                return result
                
            except Exception as e:
                self.error_counts[operation_key] = self.error_counts.get(operation_key, 0) + 1
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed, check if we should open circuit breaker
                    if self.error_counts[operation_key] >= self.circuit_breaker_threshold:
                        self.circuit_breakers[operation_key] = time.time()
                    
                    return AgentMessage(
                        id=f"error_{uuid.uuid4()}",
                        sender=self.name,
                        recipient=message.sender,
                        content=f"Failed after {self.max_retries} attempts: {str(e)}",
                        message_type="error",
                        metadata={"error_type": "max_retries_exceeded", "original_error": str(e)},
                        timestamp=time.time()
                    )
                
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
        
    async def _is_circuit_open(self, operation_key: str) -> bool:
        """Check if circuit breaker is open for given operation"""
        if operation_key not in self.circuit_breakers:
            return False
        
        # Check if timeout has passed
        if time.time() - self.circuit_breakers[operation_key] > self.circuit_breaker_timeout:
            del self.circuit_breakers[operation_key]
            return False
        
        return True
```

### 2. Monitoring and Observability
```python
# Good: Comprehensive monitoring for agent systems
import opentelemetry
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

class MonitoredAgent(BaseAgent):
    def __init__(self, name: str, role: AgentRole):
        super().__init__(name, role, [])
        self.tracer = trace.get_tracer(__name__)
        self.metrics = {
            "messages_processed": 0,
            "errors": 0,
            "average_response_time": 0.0,
            "last_activity": time.time()
        }
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process message with full observability"""
        start_time = time.time()
        
        with self.tracer.start_as_current_span(
            f"agent_{self.name}_process_message",
            attributes={
                "agent.name": self.name,
                "agent.role": self.role.value,
                "message.type": message.message_type,
                "message.sender": message.sender
            }
        ) as span:
            try:
                # Your actual message processing logic here
                result = await self._actual_process_message(message)
                
                # Update metrics
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, success=True)
                
                span.set_attribute("processing_time", processing_time)
                span.set_attribute("success", True)
                span.set_status(Status(StatusCode.OK))
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                self._update_metrics(processing_time, success=False)
                
                span.set_attribute("processing_time", processing_time)
                span.set_attribute("success", False)
                span.set_attribute("error.message", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                logger.error(f"Agent {self.name} failed to process message: {e}")
                raise
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        self.metrics["messages_processed"] += 1
        self.metrics["last_activity"] = time.time()
        
        if not success:
            self.metrics["errors"] += 1
        
        # Update running average
        current_avg = self.metrics["average_response_time"]
        message_count = self.metrics["messages_processed"]
        self.metrics["average_response_time"] = (
            (current_avg * (message_count - 1) + processing_time) / message_count
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        error_rate = self.metrics["errors"] / max(self.metrics["messages_processed"], 1)
        time_since_activity = time.time() - self.metrics["last_activity"]
        
        status = "healthy"
        if error_rate > 0.1:  # More than 10% error rate
            status = "degraded"
        if time_since_activity > 300:  # No activity for 5 minutes
            status = "inactive"
        if error_rate > 0.5:  # More than 50% error rate
            status = "unhealthy"
        
        return {
            "status": status,
            "metrics": self.metrics.copy(),
            "error_rate": error_rate,
            "time_since_last_activity": time_since_activity
        }
```

These patterns provide a robust foundation for building sophisticated AI agent systems within APES, following 2025 best practices for multi-agent coordination, error handling, and observability.
