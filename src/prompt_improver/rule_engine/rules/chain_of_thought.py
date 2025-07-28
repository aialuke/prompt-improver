"""ChainOfThoughtRule: Enhanced rule implementing step-by-step reasoning patterns.

Based on research synthesis from:
- OpenAI Chain-of-Thought research
- Zero-shot CoT techniques (Kojima et al.)
- NeurIPS CoT papers
- Structured thinking patterns for LLMs
"""

import re
from typing import Any

from ..base import BasePromptRule, RuleCheckResult, TransformationResult

# CoT trigger phrases based on research
ZERO_SHOT_COT_TRIGGERS = [
    "Let's think step by step",
    "Think through this step-by-step",
    "Let's work through this systematically",
    "Break this down step by step",
    "Let's approach this methodically",
]

# Reasoning task indicators
REASONING_TASK_PATTERNS = [
    r"\b(calculate|compute|solve|determine|find|derive)\b",
    r"\b(analyze|evaluate|assess|compare|contrast)\b",
    r"\b(explain|justify|prove|demonstrate)\b",
    r"\b(plan|design|strategy|approach)\b",
    r"\b(why|how|what if|when|where)\b",
    r"\b(because|therefore|thus|hence|consequently)\b",
]

# Complex reasoning indicators
COMPLEX_REASONING_INDICATORS = [
    "multi-step",
    "sequence",
    "process",
    "procedure",
    "algorithm",
    "logical",
    "reasoning",
    "inference",
    "deduction",
    "conclusion",
    "cause and effect",
    "relationship",
    "dependency",
    "flow",
]

class ChainOfThoughtRule(BasePromptRule):
    """Enhanced Chain of Thought rule using research-validated reasoning patterns.

    features:
    - Zero-shot CoT with optimized triggers
    - Few-shot CoT with structured examples
    - Thinking tags for intermediate reasoning
    - Step-by-step guidance with quality validation
    """

    def __init__(self):
        # Research-validated default parameters
        self.config = {
            "enable_step_by_step": True,
            "use_thinking_tags": True,
            "min_reasoning_steps": 3,
            "encourage_explicit_reasoning": True,
            "zero_shot_trigger": "Let's think step by step",
            "use_structured_response": True,
            "reasoning_quality_check": True,
            "logical_flow_validation": True,
        }

        # Attributes for dynamic loading system
        self.rule_id = "chain_of_thought"
        self.priority = 8

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {
            "name": "Chain of Thought Reasoning Rule",
            "type": "Reasoning",
            "description": "Implements step-by-step reasoning patterns based on CoT research across multiple LLM providers",
            "category": "reasoning",
            "research_foundation": [
                "OpenAI Chain-of-Thought research",
                "Zero-shot CoT techniques (Kojima et al.)",
                "NeurIPS CoT papers",
                "Structured thinking patterns",
            ],
            "version": "2.0.0",
            "priority": self.priority,
            "source": "Research Synthesis 2025",
        }

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Check if prompt would benefit from Chain of Thought reasoning"""
        reasoning_metrics = self._analyze_reasoning_requirements(prompt)

        # Determine if CoT should be applied
        applies = (
            reasoning_metrics["reasoning_complexity"] > 0.3
            or reasoning_metrics["has_multi_step_task"]
            or reasoning_metrics["logical_reasoning_needed"]
            or reasoning_metrics["problem_solving_required"]
        )

        # Don't apply if already has CoT structure
        if reasoning_metrics["already_has_cot"]:
            applies = False

        confidence = 0.9 if applies else 0.85

        return RuleCheckResult(
            applies=applies,
            confidence=confidence,
            metadata={
                "reasoning_complexity": reasoning_metrics["reasoning_complexity"],
                "has_multi_step_task": reasoning_metrics["has_multi_step_task"],
                "logical_reasoning_needed": reasoning_metrics[
                    "logical_reasoning_needed"
                ],
                "problem_solving_required": reasoning_metrics[
                    "problem_solving_required"
                ],
                "already_has_cot": reasoning_metrics["already_has_cot"],
                "reasoning_task_count": reasoning_metrics["reasoning_task_count"],
                "recommended_cot_type": reasoning_metrics["recommended_cot_type"],
                "estimated_steps": reasoning_metrics["estimated_steps"],
            },
        )

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply Chain of Thought reasoning enhancement"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(
                success=True, improved_prompt=prompt, confidence=1.0, transformations=[]
            )

        reasoning_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []

        # Apply CoT enhancements based on task complexity
        cot_type = reasoning_metrics.get("recommended_cot_type", "zero_shot")

        if cot_type == "zero_shot":
            improved_prompt, zero_shot_transformations = self._apply_zero_shot_cot(
                improved_prompt
            )
            transformations.extend(zero_shot_transformations)
        elif cot_type == "few_shot":
            improved_prompt, few_shot_transformations = self._apply_few_shot_cot(
                improved_prompt, reasoning_metrics
            )
            transformations.extend(few_shot_transformations)
        elif cot_type == "structured":
            improved_prompt, structured_transformations = self._apply_structured_cot(
                improved_prompt, reasoning_metrics
            )
            transformations.extend(structured_transformations)

        # Add thinking tags if enabled
        if self.config["use_thinking_tags"]:
            improved_prompt, thinking_transformations = self._add_thinking_structure(
                improved_prompt
            )
            transformations.extend(thinking_transformations)

        # Add step-by-step guidance if requested
        if self.config["enable_step_by_step"]:
            improved_prompt, step_transformations = self._add_step_guidance(
                improved_prompt, reasoning_metrics
            )
            transformations.extend(step_transformations)

        # Calculate confidence based on improvements
        confidence = min(0.95, 0.7 + (len(transformations) * 0.05))

        return TransformationResult(
            success=True,
            improved_prompt=improved_prompt,
            confidence=confidence,
            transformations=transformations,
        )

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for CoT reasoning"""
        return """
<instruction>
Apply Chain of Thought reasoning using research-validated patterns:

1. ZERO-SHOT COT (Kojima et al.):
   - Add "Let's think step by step" for complex reasoning tasks
   - Encourage explicit intermediate steps
   - Break down multi-part problems systematically

2. STRUCTURED THINKING:
   - Use <thinking> tags for intermediate reasoning
   - Show logical progression from premises to conclusions
   - Validate each step before proceeding

3. STEP-BY-STEP GUIDANCE:
   - Number reasoning steps clearly (1, 2, 3...)
   - Connect each step to the next logically
   - Summarize key insights at each stage

4. REASONING QUALITY:
   - Verify logical consistency between steps
   - Check for missing assumptions or gaps
   - Ensure conclusion follows from premises

Focus on making the reasoning process explicit, traceable, and verifiable.
</instruction>
"""

    def _analyze_reasoning_requirements(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine CoT reasoning requirements"""
        words = prompt.lower().split()

        # Check for reasoning task patterns
        reasoning_task_count = sum(
            len(re.findall(pattern, prompt, re.IGNORECASE))
            for pattern in REASONING_TASK_PATTERNS
        )

        # Check for complex reasoning indicators
        complex_indicators = sum(
            1
            for indicator in COMPLEX_REASONING_INDICATORS
            if indicator in prompt.lower()
        )

        # Check for multi-step task indicators
        multi_step_words = [
            "then",
            "next",
            "after",
            "following",
            "subsequently",
            "first",
            "second",
            "finally",
        ]
        has_multi_step_task = any(word in words for word in multi_step_words)

        # Check for logical reasoning needs
        logical_words = [
            "because",
            "therefore",
            "thus",
            "hence",
            "since",
            "given that",
            "if",
            "unless",
        ]
        logical_reasoning_needed = any(
            phrase in prompt.lower() for phrase in logical_words
        )

        # Check for problem-solving requirements
        problem_words = ["solve", "calculate", "determine", "find", "compute", "derive"]
        problem_solving_required = any(word in words for word in problem_words)

        # Check if already has CoT structure
        cot_indicators = [
            "step by step",
            "thinking",
            "reasoning",
            "let's think",
            "first,",
            "second,",
            "third,",
        ]
        already_has_cot = any(
            indicator in prompt.lower() for indicator in cot_indicators
        )

        # Calculate reasoning complexity score
        reasoning_complexity = (
            (reasoning_task_count / max(len(words), 1)) * 0.4
            + (complex_indicators / max(len(words), 1)) * 0.3
            + (1.0 if has_multi_step_task else 0.0) * 0.2
            + (1.0 if logical_reasoning_needed else 0.0) * 0.1
        )

        # Determine recommended CoT type
        if reasoning_complexity > 0.7:
            recommended_cot_type = "structured"
        elif reasoning_complexity > 0.4 or has_multi_step_task:
            recommended_cot_type = "few_shot"
        else:
            recommended_cot_type = "zero_shot"

        # Estimate number of reasoning steps needed
        estimated_steps = max(
            self.config["min_reasoning_steps"],
            min(8, reasoning_task_count + (2 if has_multi_step_task else 0)),
        )

        return {
            "reasoning_complexity": reasoning_complexity,
            "has_multi_step_task": has_multi_step_task,
            "logical_reasoning_needed": logical_reasoning_needed,
            "problem_solving_required": problem_solving_required,
            "already_has_cot": already_has_cot,
            "reasoning_task_count": reasoning_task_count,
            "complex_indicators": complex_indicators,
            "recommended_cot_type": recommended_cot_type,
            "estimated_steps": estimated_steps,
        }

    def _apply_zero_shot_cot(self, prompt: str) -> tuple[str, list[dict]]:
        """Apply zero-shot CoT trigger based on Kojima et al. research"""
        trigger = self.config["zero_shot_trigger"]

        # Add the trigger at the end of the prompt
        enhanced_prompt = f"{prompt}\n\n{trigger}."

        return enhanced_prompt, [
            {
                "type": "zero_shot_cot",
                "description": f"Added zero-shot CoT trigger: '{trigger}'",
                "research_basis": "Kojima et al. zero-shot CoT techniques",
            }
        ]

    def _apply_few_shot_cot(
        self, prompt: str, reasoning_metrics: dict
    ) -> tuple[str, list[dict]]:
        """Apply few-shot CoT with structured examples"""
        # Generate appropriate examples based on task type
        task_type = self._identify_task_type(prompt)
        example = self._get_cot_example(task_type)

        enhanced_prompt = f"""Here's an example of step-by-step reasoning:

{example}

Now, apply the same step-by-step approach to: {prompt}"""

        return enhanced_prompt, [
            {
                "type": "few_shot_cot",
                "description": f"Added CoT example for {task_type} task",
                "research_basis": "Few-shot CoT with structured examples",
            }
        ]

    def _apply_structured_cot(
        self, prompt: str, reasoning_metrics: dict
    ) -> tuple[str, list[dict]]:
        """Apply structured CoT with explicit step framework"""
        estimated_steps = reasoning_metrics.get("estimated_steps", 3)

        step_framework = "\n".join([
            f"Step {i}: [Your reasoning for step {i}]"
            for i in range(1, estimated_steps + 1)
        ])

        enhanced_prompt = f"""{prompt}

Please work through this systematically:

{step_framework}

Final Answer: [Your conclusion based on the steps above]"""

        return enhanced_prompt, [
            {
                "type": "structured_cot",
                "description": f"Added structured framework with {estimated_steps} reasoning steps",
                "research_basis": "Structured thinking patterns for complex reasoning",
            }
        ]

    def _add_thinking_structure(self, prompt: str) -> tuple[str, list[dict]]:
        """Add thinking tags for intermediate reasoning"""
        if "<thinking>" in prompt.lower():
            return prompt, []  # Already has thinking structure

        enhanced_prompt = f"""{prompt}

<thinking>
Let me break this down step by step:
1. First, I need to understand what is being asked...
2. Then, I should consider the key factors...
3. Next, I'll work through the logical steps...
4. Finally, I'll arrive at a well-reasoned conclusion...
</thinking>

<response>
[Your final answer here]
</response>"""

        return enhanced_prompt, [
            {
                "type": "thinking_structure",
                "description": "Added thinking tags for explicit intermediate reasoning",
                "research_basis": "Structured thinking patterns for LLMs",
            }
        ]

    def _add_step_guidance(
        self, prompt: str, reasoning_metrics: dict
    ) -> tuple[str, list[dict]]:
        """Add step-by-step guidance based on task complexity"""
        if "step" in prompt.lower():
            return prompt, []  # Already has step guidance

        guidance = "\n\nPlease approach this step-by-step:\n1. Identify the key components\n2. Analyze the relationships\n3. Work through the logic systematically\n4. Verify your reasoning\n5. Provide a clear conclusion"

        enhanced_prompt = prompt + guidance

        return enhanced_prompt, [
            {
                "type": "step_guidance",
                "description": "Added explicit step-by-step guidance framework",
                "research_basis": "Systematic reasoning methodology",
            }
        ]

    def _identify_task_type(self, prompt: str) -> str:
        """Identify the type of reasoning task for appropriate examples"""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower for word in ["calculate", "compute", "math", "number"]
        ):
            return "mathematical"
        if any(word in prompt_lower for word in ["analyze", "evaluate", "compare"]):
            return "analytical"
        if any(word in prompt_lower for word in ["solve", "problem", "solution"]):
            return "problem_solving"
        if any(word in prompt_lower for word in ["plan", "design", "strategy"]):
            return "planning"
        return "general"

    def _get_cot_example(self, task_type: str) -> str:
        """Get appropriate CoT example based on task type"""
        examples = {
            "mathematical": """
Problem: If a store sells 3 apples for $2, how much would 12 apples cost?

Thinking step by step:
1. First, I need to find the cost per apple: $2 ÷ 3 apples = $0.67 per apple
2. Then, I calculate the cost for 12 apples: $0.67 × 12 = $8.04
3. Therefore, 12 apples would cost $8.04
""",
            "analytical": """
Question: What are the advantages and disadvantages of remote work?

Thinking step by step:
1. First, I'll identify the main advantages:
   - Flexibility in work schedule and location
   - Reduced commuting time and costs
   - Better work-life balance potential
2. Next, I'll consider the disadvantages:
   - Potential isolation and reduced collaboration
   - Distractions at home
   - Communication challenges
3. Finally, I'll synthesize: Remote work offers significant benefits but requires careful management of its challenges
""",
            "problem_solving": """
Problem: How can a company reduce customer churn?

Thinking step by step:
1. First, identify why customers leave:
   - Poor customer service
   - Better competitor offers
   - Unmet needs
2. Then, develop targeted solutions:
   - Improve customer support training
   - Enhance product features
   - Implement loyalty programs
3. Finally, implement measurement systems to track improvement
""",
            "planning": """
Task: Plan a marketing campaign for a new product

Thinking step by step:
1. First, define the target audience and their needs
2. Then, determine the key messaging and value proposition
3. Next, select appropriate marketing channels and budget
4. Create a timeline with milestones and success metrics
5. Finally, plan for monitoring and optimization during execution
""",
            "general": """
Question: How should I approach this complex task?

Thinking step by step:
1. First, I'll break down the task into smaller components
2. Then, I'll analyze each component and its requirements
3. Next, I'll identify dependencies and logical sequence
4. Finally, I'll synthesize a comprehensive approach
""",
        }

        return examples.get(task_type, examples["general"])
