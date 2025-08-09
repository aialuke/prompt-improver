"""ChainOfThoughtRule: Enhanced rule implementing step-by-step reasoning patterns.

Based on research synthesis from:
- OpenAI Chain-of-Thought research
- Zero-shot CoT techniques (Kojima et al.)
- NeurIPS CoT papers
- Structured thinking patterns for LLMs
"""
import re
from typing import Any
from prompt_improver.rule_engine.base import BasePromptRule, RuleCheckResult, TransformationResult
ZERO_SHOT_COT_TRIGGERS = ["Let's think step by step", 'Think through this step-by-step', "Let's work through this systematically", 'Break this down step by step', "Let's approach this methodically"]
REASONING_TASK_PATTERNS = ['\\b(calculate|compute|solve|determine|find|derive)\\b', '\\b(analyze|evaluate|assess|compare|contrast)\\b', '\\b(explain|justify|prove|demonstrate)\\b', '\\b(plan|design|strategy|approach)\\b', '\\b(why|how|what if|when|where)\\b', '\\b(because|therefore|thus|hence|consequently)\\b']
COMPLEX_REASONING_INDICATORS = ['multi-step', 'sequence', 'process', 'procedure', 'algorithm', 'logical', 'reasoning', 'inference', 'deduction', 'conclusion', 'cause and effect', 'relationship', 'dependency', 'flow']

class ChainOfThoughtRule(BasePromptRule):
    """Enhanced Chain of Thought rule using research-validated reasoning patterns.

    features:
    - Zero-shot CoT with optimized triggers
    - Few-shot CoT with structured examples
    - Thinking tags for intermediate reasoning
    - Step-by-step guidance with quality validation
    """

    def __init__(self):
        self.config = {'enable_step_by_step': True, 'use_thinking_tags': True, 'min_reasoning_steps': 3, 'encourage_explicit_reasoning': True, 'zero_shot_trigger': "Let's think step by step", 'use_structured_response': True, 'reasoning_quality_check': True, 'logical_flow_validation': True}
        self.rule_id = 'chain_of_thought'
        self.priority = 8

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {'name': 'Chain of Thought Reasoning Rule', 'type': 'Reasoning', 'description': 'Implements step-by-step reasoning patterns based on CoT research across multiple LLM providers', 'category': 'reasoning', 'research_foundation': ['OpenAI Chain-of-Thought research', 'Zero-shot CoT techniques (Kojima et al.)', 'NeurIPS CoT papers', 'Structured thinking patterns'], 'version': '2.0.0', 'priority': self.priority, 'source': 'Research Synthesis 2025'}

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Check if prompt would benefit from Chain of Thought reasoning"""
        reasoning_metrics = self._analyze_reasoning_requirements(prompt)
        applies = reasoning_metrics['reasoning_complexity'] > 0.3 or reasoning_metrics['has_multi_step_task'] or reasoning_metrics['logical_reasoning_needed'] or reasoning_metrics['problem_solving_required']
        if reasoning_metrics['already_has_cot']:
            applies = False
        confidence = 0.9 if applies else 0.85
        return RuleCheckResult(applies=applies, confidence=confidence, metadata={'reasoning_complexity': reasoning_metrics['reasoning_complexity'], 'has_multi_step_task': reasoning_metrics['has_multi_step_task'], 'logical_reasoning_needed': reasoning_metrics['logical_reasoning_needed'], 'problem_solving_required': reasoning_metrics['problem_solving_required'], 'already_has_cot': reasoning_metrics['already_has_cot'], 'reasoning_task_count': reasoning_metrics['reasoning_task_count'], 'recommended_cot_type': reasoning_metrics['recommended_cot_type'], 'estimated_steps': reasoning_metrics['estimated_steps']})

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply Chain of Thought reasoning enhancement"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(success=True, improved_prompt=prompt, confidence=1.0, transformations=[])
        reasoning_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []
        cot_type = reasoning_metrics.get('recommended_cot_type', 'zero_shot')
        if cot_type == 'zero_shot':
            improved_prompt, zero_shot_transformations = self._apply_zero_shot_cot(improved_prompt)
            transformations.extend(zero_shot_transformations)
        elif cot_type == 'few_shot':
            improved_prompt, few_shot_transformations = self._apply_few_shot_cot(improved_prompt, reasoning_metrics)
            transformations.extend(few_shot_transformations)
        elif cot_type == 'structured':
            improved_prompt, structured_transformations = self._apply_structured_cot(improved_prompt, reasoning_metrics)
            transformations.extend(structured_transformations)
        if self.config['use_thinking_tags']:
            improved_prompt, thinking_transformations = self._add_thinking_structure(improved_prompt)
            transformations.extend(thinking_transformations)
        if self.config['enable_step_by_step']:
            improved_prompt, step_transformations = self._add_step_guidance(improved_prompt, reasoning_metrics)
            transformations.extend(step_transformations)
        confidence = min(0.95, 0.7 + len(transformations) * 0.05)
        return TransformationResult(success=True, improved_prompt=improved_prompt, confidence=confidence, transformations=transformations)

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for CoT reasoning"""
        return '\n<instruction>\nApply Chain of Thought reasoning using research-validated patterns:\n\n1. ZERO-SHOT COT (Kojima et al.):\n   - Add "Let\'s think step by step" for complex reasoning tasks\n   - Encourage explicit intermediate steps\n   - Break down multi-part problems systematically\n\n2. STRUCTURED THINKING:\n   - Use <thinking> tags for intermediate reasoning\n   - Show logical progression from premises to conclusions\n   - Validate each step before proceeding\n\n3. STEP-BY-STEP GUIDANCE:\n   - Number reasoning steps clearly (1, 2, 3...)\n   - Connect each step to the next logically\n   - Summarize key insights at each stage\n\n4. REASONING QUALITY:\n   - Verify logical consistency between steps\n   - Check for missing assumptions or gaps\n   - Ensure conclusion follows from premises\n\nFocus on making the reasoning process explicit, traceable, and verifiable.\n</instruction>\n'

    def _analyze_reasoning_requirements(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine CoT reasoning requirements"""
        words = prompt.lower().split()
        reasoning_task_count = sum((len(re.findall(pattern, prompt, re.IGNORECASE)) for pattern in REASONING_TASK_PATTERNS))
        complex_indicators = sum((1 for indicator in COMPLEX_REASONING_INDICATORS if indicator in prompt.lower()))
        multi_step_words = ['then', 'next', 'after', 'following', 'subsequently', 'first', 'second', 'finally']
        has_multi_step_task = any((word in words for word in multi_step_words))
        logical_words = ['because', 'therefore', 'thus', 'hence', 'since', 'given that', 'if', 'unless']
        logical_reasoning_needed = any((phrase in prompt.lower() for phrase in logical_words))
        problem_words = ['solve', 'calculate', 'determine', 'find', 'compute', 'derive']
        problem_solving_required = any((word in words for word in problem_words))
        cot_indicators = ['step by step', 'thinking', 'reasoning', "let's think", 'first,', 'second,', 'third,']
        already_has_cot = any((indicator in prompt.lower() for indicator in cot_indicators))
        reasoning_complexity = reasoning_task_count / max(len(words), 1) * 0.4 + complex_indicators / max(len(words), 1) * 0.3 + (1.0 if has_multi_step_task else 0.0) * 0.2 + (1.0 if logical_reasoning_needed else 0.0) * 0.1
        if reasoning_complexity > 0.7:
            recommended_cot_type = 'structured'
        elif reasoning_complexity > 0.4 or has_multi_step_task:
            recommended_cot_type = 'few_shot'
        else:
            recommended_cot_type = 'zero_shot'
        estimated_steps = max(self.config['min_reasoning_steps'], min(8, reasoning_task_count + (2 if has_multi_step_task else 0)))
        return {'reasoning_complexity': reasoning_complexity, 'has_multi_step_task': has_multi_step_task, 'logical_reasoning_needed': logical_reasoning_needed, 'problem_solving_required': problem_solving_required, 'already_has_cot': already_has_cot, 'reasoning_task_count': reasoning_task_count, 'complex_indicators': complex_indicators, 'recommended_cot_type': recommended_cot_type, 'estimated_steps': estimated_steps}

    def _apply_zero_shot_cot(self, prompt: str) -> tuple[str, list[dict]]:
        """Apply zero-shot CoT trigger based on Kojima et al. research"""
        trigger = self.config['zero_shot_trigger']
        enhanced_prompt = f'{prompt}\n\n{trigger}.'
        return (enhanced_prompt, [{'type': 'zero_shot_cot', 'description': f"Added zero-shot CoT trigger: '{trigger}'", 'research_basis': 'Kojima et al. zero-shot CoT techniques'}])

    def _apply_few_shot_cot(self, prompt: str, reasoning_metrics: dict) -> tuple[str, list[dict]]:
        """Apply few-shot CoT with structured examples"""
        task_type = self._identify_task_type(prompt)
        example = self._get_cot_example(task_type)
        enhanced_prompt = f"Here's an example of step-by-step reasoning:\n\n{example}\n\nNow, apply the same step-by-step approach to: {prompt}"
        return (enhanced_prompt, [{'type': 'few_shot_cot', 'description': f'Added CoT example for {task_type} task', 'research_basis': 'Few-shot CoT with structured examples'}])

    def _apply_structured_cot(self, prompt: str, reasoning_metrics: dict) -> tuple[str, list[dict]]:
        """Apply structured CoT with explicit step framework"""
        estimated_steps = reasoning_metrics.get('estimated_steps', 3)
        step_framework = '\n'.join([f'Step {i}: [Your reasoning for step {i}]' for i in range(1, estimated_steps + 1)])
        enhanced_prompt = f'{prompt}\n\nPlease work through this systematically:\n\n{step_framework}\n\nFinal Answer: [Your conclusion based on the steps above]'
        return (enhanced_prompt, [{'type': 'structured_cot', 'description': f'Added structured framework with {estimated_steps} reasoning steps', 'research_basis': 'Structured thinking patterns for complex reasoning'}])

    def _add_thinking_structure(self, prompt: str) -> tuple[str, list[dict]]:
        """Add thinking tags for intermediate reasoning"""
        if '<thinking>' in prompt.lower():
            return (prompt, [])
        enhanced_prompt = f"{prompt}\n\n<thinking>\nLet me break this down step by step:\n1. First, I need to understand what is being asked...\n2. Then, I should consider the key factors...\n3. Next, I'll work through the logical steps...\n4. Finally, I'll arrive at a well-reasoned conclusion...\n</thinking>\n\n<response>\n[Your final answer here]\n</response>"
        return (enhanced_prompt, [{'type': 'thinking_structure', 'description': 'Added thinking tags for explicit intermediate reasoning', 'research_basis': 'Structured thinking patterns for LLMs'}])

    def _add_step_guidance(self, prompt: str, reasoning_metrics: dict) -> tuple[str, list[dict]]:
        """Add step-by-step guidance based on task complexity"""
        if 'step' in prompt.lower():
            return (prompt, [])
        guidance = '\n\nPlease approach this step-by-step:\n1. Identify the key components\n2. Analyze the relationships\n3. Work through the logic systematically\n4. Verify your reasoning\n5. Provide a clear conclusion'
        enhanced_prompt = prompt + guidance
        return (enhanced_prompt, [{'type': 'step_guidance', 'description': 'Added explicit step-by-step guidance framework', 'research_basis': 'Systematic reasoning methodology'}])

    def _identify_task_type(self, prompt: str) -> str:
        """Identify the type of reasoning task for appropriate examples"""
        prompt_lower = prompt.lower()
        if any((word in prompt_lower for word in ['calculate', 'compute', 'math', 'number'])):
            return 'mathematical'
        if any((word in prompt_lower for word in ['analyze', 'evaluate', 'compare'])):
            return 'analytical'
        if any((word in prompt_lower for word in ['solve', 'problem', 'solution'])):
            return 'problem_solving'
        if any((word in prompt_lower for word in ['plan', 'design', 'strategy'])):
            return 'planning'
        return 'general'

    def _get_cot_example(self, task_type: str) -> str:
        """Get appropriate CoT example based on task type"""
        examples = {'mathematical': '\nProblem: If a store sells 3 apples for $2, how much would 12 apples cost?\n\nThinking step by step:\n1. First, I need to find the cost per apple: $2 ÷ 3 apples = $0.67 per apple\n2. Then, I calculate the cost for 12 apples: $0.67 × 12 = $8.04\n3. Therefore, 12 apples would cost $8.04\n', 'analytical': "\nQuestion: What are the advantages and disadvantages of remote work?\n\nThinking step by step:\n1. First, I'll identify the main advantages:\n   - Flexibility in work schedule and location\n   - Reduced commuting time and costs\n   - Better work-life balance potential\n2. Next, I'll consider the disadvantages:\n   - Potential isolation and reduced collaboration\n   - Distractions at home\n   - Communication challenges\n3. Finally, I'll synthesize: Remote work offers significant benefits but requires careful management of its challenges\n", 'problem_solving': '\nProblem: How can a company reduce customer churn?\n\nThinking step by step:\n1. First, identify why customers leave:\n   - Poor customer service\n   - Better competitor offers\n   - Unmet needs\n2. Then, develop targeted solutions:\n   - Improve customer support training\n   - Enhance product features\n   - Implement loyalty programs\n3. Finally, implement measurement systems to track improvement\n', 'planning': '\nTask: Plan a marketing campaign for a new product\n\nThinking step by step:\n1. First, define the target audience and their needs\n2. Then, determine the key messaging and value proposition\n3. Next, select appropriate marketing channels and budget\n4. Create a timeline with milestones and success metrics\n5. Finally, plan for monitoring and optimization during execution\n', 'general': "\nQuestion: How should I approach this complex task?\n\nThinking step by step:\n1. First, I'll break down the task into smaller components\n2. Then, I'll analyze each component and its requirements\n3. Next, I'll identify dependencies and logical sequence\n4. Finally, I'll synthesize a comprehensive approach\n"}
        return examples.get(task_type, examples['general'])
