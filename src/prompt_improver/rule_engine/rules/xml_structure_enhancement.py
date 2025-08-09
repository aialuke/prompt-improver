"""XMLStructureRule: Enhanced rule for XML structuring based on Anthropic optimization patterns.

Based on research synthesis from:
- Anthropic XML optimization guide
- Claude-specific structuring patterns for improved performance
- Structured prompt organization best practices
- XML hierarchy and tagging effectiveness studies
"""
import re
from typing import Any
from prompt_improver.rule_engine.base import BasePromptRule, RuleCheckResult, TransformationResult
RECOMMENDED_XML_TAGS = {'context': ['background', 'situation', 'given', 'context', 'scenario'], 'instruction': ['task', 'instruction', 'request', 'objective', 'goal'], 'examples': ['example', 'sample', 'illustration', 'demonstration'], 'thinking': ['reasoning', 'analysis', 'thought', 'consideration'], 'response': ['answer', 'output', 'result', 'solution'], 'format': ['format', 'structure', 'template', 'layout'], 'constraints': ['constraint', 'limitation', 'requirement', 'rule'], 'criteria': ['criteria', 'evaluation', 'success', 'metric']}
COMPLEXITY_INDICATORS = ['multiple', 'several', 'various', 'different', 'complex', 'detailed', 'comprehensive', 'thorough', 'step-by-step', 'structured', 'organized']
CONTENT_PATTERNS = {'context': ['given that', 'in the context of', 'background', 'situation', 'considering', 'based on', 'scenario', 'setting'], 'instruction': ['please', 'can you', 'I need', 'help me', 'create', 'write', 'generate', 'analyze', 'evaluate', 'explain', 'describe'], 'examples': ['for example', 'such as', 'like', 'including', 'e\\.g\\.', 'instance', 'sample', 'illustration'], 'constraints': ['must', 'should', 'cannot', 'do not', 'avoid', 'ensure', 'requirement', 'constraint', 'limitation', 'restriction'], 'format': ['format', 'structure', 'organize', 'layout', 'arrange', 'present as', 'output as', 'in the form of']}

class XMLStructureRule(BasePromptRule):
    """Enhanced XML structure rule using Anthropic optimization patterns.

    features:
    - Context, instruction, example, thinking, and response tag organization
    - Automatic content type detection and appropriate tagging
    - Nested structure support with hierarchy enforcement
    - Minimal attribute usage for optimal Claude performance
    """

    def __init__(self):
        self.config = {'use_context_tags': True, 'use_instruction_tags': True, 'use_example_tags': True, 'use_thinking_tags': True, 'use_response_tags': True, 'nested_structure_allowed': True, 'attribute_usage': 'minimal', 'tag_hierarchy_enforcement': True}
        self.rule_id = 'xml_structure_enhancement'
        self.priority = 5

    def configure(self, params: dict[str, Any]):
        """Configure rule parameters from database"""
        self.config.update(params)

    @property
    def metadata(self):
        """Enhanced metadata with research foundation"""
        return {'name': 'XML Structure Enhancement Rule', 'type': 'Structure', 'description': 'Implements XML tagging patterns recommended by Anthropic for Claude optimization', 'category': 'structure', 'research_foundation': ['Anthropic XML optimization guide', 'Claude-specific structuring patterns', 'Structured prompt organization'], 'version': '2.0.0', 'priority': self.priority, 'source': 'Research Synthesis 2025'}

    def check(self, prompt: str, context=None) -> RuleCheckResult:
        """Check if prompt would benefit from XML structure"""
        structure_metrics = self._analyze_structure_requirements(prompt)
        applies = structure_metrics['complexity_score'] > 0.4 and (not structure_metrics['already_has_xml']) and structure_metrics['benefits_from_structure'] and (structure_metrics['word_count'] > 20)
        confidence = 0.9 if applies else 0.85
        return RuleCheckResult(applies=applies, confidence=confidence, metadata={'complexity_score': structure_metrics['complexity_score'], 'word_count': structure_metrics['word_count'], 'already_has_xml': structure_metrics['already_has_xml'], 'benefits_from_structure': structure_metrics['benefits_from_structure'], 'detected_sections': structure_metrics['detected_sections'], 'recommended_tags': structure_metrics['recommended_tags'], 'structure_type': structure_metrics['structure_type']})

    def apply(self, prompt: str, context=None) -> TransformationResult:
        """Apply XML structure enhancement"""
        check_result = self.check(prompt, context)
        if not check_result.applies:
            return TransformationResult(success=True, improved_prompt=prompt, confidence=1.0, transformations=[])
        structure_metrics = check_result.metadata
        improved_prompt = prompt
        transformations = []
        detected_sections = structure_metrics.get('detected_sections', {})
        if detected_sections:
            improved_prompt, xml_transformations = self._apply_xml_structure(improved_prompt, detected_sections, structure_metrics)
            transformations.extend(xml_transformations)
        else:
            improved_prompt, basic_transformations = self._apply_basic_xml_structure(improved_prompt, structure_metrics)
            transformations.extend(basic_transformations)
        confidence = min(0.95, 0.75 + len(transformations) * 0.05)
        return TransformationResult(success=True, improved_prompt=improved_prompt, confidence=confidence, transformations=transformations)

    def to_llm_instruction(self) -> str:
        """Generate research-based LLM instruction for XML structure"""
        return '\n<instruction>\nApply XML structure using Anthropic optimization patterns:\n\n1. CORE TAGS (Anthropic Recommended):\n   - <context>: Background information and setting\n   - <instruction>: Main task or request\n   - <examples>: Sample inputs/outputs or demonstrations\n   - <thinking>: Reasoning process or analysis\n   - <response>: Final answer or output\n\n2. ORGANIZATION PRINCIPLES:\n   - Place context before instructions (critical information first)\n   - Group related information within appropriate tags\n   - Use clear, semantic tag names\n   - Maintain logical hierarchy\n\n3. ATTRIBUTE USAGE:\n   - Minimize attributes for optimal Claude performance\n   - Use only when necessary for clarity\n   - Prefer semantic tag names over attributes\n\n4. NESTED STRUCTURE:\n   - Allow nesting when content naturally groups\n   - Maintain consistent indentation\n   - Ensure proper tag closure\n\nFocus on clarity and logical organization that enhances understanding.\n</instruction>\n'

    def _analyze_structure_requirements(self, prompt: str) -> dict[str, Any]:
        """Analyze prompt to determine XML structure requirements"""
        words = prompt.split()
        word_count = len(words)
        xml_pattern = '<[^>]+>'
        already_has_xml = bool(re.search(xml_pattern, prompt))
        complexity_score = self._calculate_complexity_score(prompt)
        detected_sections = self._detect_content_sections(prompt)
        benefits_from_structure = word_count > 20 and (complexity_score > 0.3 or len(detected_sections) > 1)
        recommended_tags = self._recommend_tags(detected_sections, complexity_score)
        if len(detected_sections) > 3:
            structure_type = 'comprehensive'
        elif len(detected_sections) > 1:
            structure_type = 'moderate'
        else:
            structure_type = 'basic'
        return {'complexity_score': complexity_score, 'word_count': word_count, 'already_has_xml': already_has_xml, 'benefits_from_structure': benefits_from_structure, 'detected_sections': detected_sections, 'recommended_tags': recommended_tags, 'structure_type': structure_type}

    def _calculate_complexity_score(self, prompt: str) -> float:
        """Calculate prompt complexity to determine XML structure need"""
        words = prompt.lower().split()
        complexity_count = sum((1 for word in words if word in COMPLEXITY_INDICATORS))
        sentence_count = len(re.findall('[.!?]+', prompt))
        conjunctions = ['and', 'but', 'however', 'therefore', 'furthermore', 'moreover', 'additionally']
        conjunction_count = sum((1 for word in words if word in conjunctions))
        base_complexity = complexity_count / max(len(words), 1) * 5
        sentence_complexity = min(0.3, sentence_count / 10)
        conjunction_complexity = conjunction_count / max(len(words), 1) * 3
        length_complexity = min(0.4, len(words) / 50)
        total_complexity = min(1.0, base_complexity + sentence_complexity + conjunction_complexity + length_complexity)
        return total_complexity

    def _detect_content_sections(self, prompt: str) -> dict[str, str]:
        """Detect different content sections in the prompt"""
        sections = {}
        prompt_lower = prompt.lower()
        sentences = re.split('[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        context_sentences = []
        instruction_sentences = []
        example_sentences = []
        constraint_sentences = []
        format_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any((re.search(pattern, sentence_lower) for pattern in CONTENT_PATTERNS['context'])):
                context_sentences.append(sentence)
            elif any((re.search(pattern, sentence_lower) for pattern in CONTENT_PATTERNS['instruction'])):
                instruction_sentences.append(sentence)
            elif any((re.search(pattern, sentence_lower) for pattern in CONTENT_PATTERNS['examples'])):
                example_sentences.append(sentence)
            elif any((re.search(pattern, sentence_lower) for pattern in CONTENT_PATTERNS['constraints'])):
                constraint_sentences.append(sentence)
            elif any((re.search(pattern, sentence_lower) for pattern in CONTENT_PATTERNS['format'])):
                format_sentences.append(sentence)
            else:
                instruction_sentences.append(sentence)
        if context_sentences:
            sections['context'] = '. '.join(context_sentences) + '.'
        if instruction_sentences:
            sections['instruction'] = '. '.join(instruction_sentences) + '.'
        if example_sentences:
            sections['examples'] = '. '.join(example_sentences) + '.'
        if constraint_sentences:
            sections['constraints'] = '. '.join(constraint_sentences) + '.'
        if format_sentences:
            sections['format'] = '. '.join(format_sentences) + '.'
        return sections

    def _recommend_tags(self, detected_sections: dict[str, str], complexity_score: float) -> list[str]:
        """Recommend appropriate XML tags based on content and complexity"""
        recommended = []
        if 'context' in detected_sections and self.config['use_context_tags']:
            recommended.append('context')
        if 'instruction' in detected_sections and self.config['use_instruction_tags']:
            recommended.append('instruction')
        if 'examples' in detected_sections and self.config['use_example_tags']:
            recommended.append('examples')
        if 'format' in detected_sections:
            recommended.append('format')
        if 'constraints' in detected_sections:
            recommended.append('constraints')
        if complexity_score > 0.6 and self.config['use_thinking_tags']:
            recommended.append('thinking')
        if complexity_score > 0.5 and self.config['use_response_tags']:
            recommended.append('response')
        return recommended

    def _apply_xml_structure(self, prompt: str, detected_sections: dict[str, str], metrics: dict) -> tuple[str, list[dict]]:
        """Apply comprehensive XML structure based on detected sections"""
        structured_parts = []
        transformations = []
        tag_order = ['context', 'examples', 'instruction', 'format', 'constraints']
        for tag in tag_order:
            if tag in detected_sections and self.config.get(f'use_{tag}_tags', True):
                content = detected_sections[tag]
                structured_parts.append(f'<{tag}>\n{content}\n</{tag}>')
                transformations.append({'type': 'xml_structure_enhancement', 'description': f'Added <{tag}> tags for content organization', 'tag': tag, 'research_basis': 'Anthropic XML optimization patterns'})
        if metrics.get('complexity_score', 0) > 0.6 and self.config['use_thinking_tags']:
            thinking_template = '<thinking>\nLet me break this down systematically:\n1. Understanding the requirements...\n2. Analyzing the key components...\n3. Developing the approach...\n</thinking>'
            structured_parts.append(thinking_template)
            transformations.append({'type': 'xml_structure_enhancement', 'description': 'Added <thinking> tags for structured reasoning', 'tag': 'thinking', 'research_basis': 'Anthropic structured thinking patterns'})
        if self.config['use_response_tags']:
            structured_parts.append('<response>\n[Your answer here]\n</response>')
            transformations.append({'type': 'xml_structure_enhancement', 'description': 'Added <response> tags for clear output format', 'tag': 'response', 'research_basis': 'Anthropic response structuring'})
        if structured_parts:
            structured_prompt = '\n\n'.join(structured_parts)
            return (structured_prompt, transformations)
        return (prompt, [])

    def _apply_basic_xml_structure(self, prompt: str, metrics: dict) -> tuple[str, list[dict]]:
        """Apply basic XML structure for prompts without clear sections"""
        if metrics.get('word_count', 0) < 30:
            return (prompt, [])
        transformations = []
        sentences = re.split('[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < 2:
            structured_prompt = f'<instruction>\n{prompt}\n</instruction>'
            transformations.append({'type': 'xml_structure_enhancement', 'description': 'Added basic <instruction> tags for simple prompt', 'tag': 'instruction', 'research_basis': 'Anthropic basic structuring'})
        else:
            midpoint = max(1, len(sentences) // 2)
            context_part = '. '.join(sentences[:midpoint]) + '.'
            instruction_part = '. '.join(sentences[midpoint:]) + '.'
            structured_parts = []
            if self.config['use_context_tags'] and context_part:
                structured_parts.append(f'<context>\n{context_part}\n</context>')
                transformations.append({'type': 'xml_structure_enhancement', 'description': 'Added <context> tags for background information', 'tag': 'context', 'research_basis': 'Anthropic context organization'})
            if self.config['use_instruction_tags'] and instruction_part:
                structured_parts.append(f'<instruction>\n{instruction_part}\n</instruction>')
                transformations.append({'type': 'xml_structure_enhancement', 'description': 'Added <instruction> tags for task specification', 'tag': 'instruction', 'research_basis': 'Anthropic instruction structuring'})
            structured_prompt = '\n\n'.join(structured_parts) if structured_parts else prompt
        return (structured_prompt, transformations)

    def _validate_xml_structure(self, structured_prompt: str) -> bool:
        """Validate that the XML structure is well-formed"""
        try:
            tag_pattern = '<(\\w+)>'
            opening_tags = re.findall(tag_pattern, structured_prompt)
            closing_pattern = '</(\\w+)>'
            closing_tags = re.findall(closing_pattern, structured_prompt)
            return sorted(opening_tags) == sorted(closing_tags)
        except Exception:
            return False

    def _optimize_tag_hierarchy(self, sections: dict[str, str]) -> dict[str, str]:
        """Optimize tag hierarchy based on Anthropic best practices"""
        optimized_order = ['context', 'examples', 'instruction', 'constraints', 'format']
        optimized_sections = {}
        for tag in optimized_order:
            if tag in sections:
                optimized_sections[tag] = sections[tag]
        for tag, content in sections.items():
            if tag not in optimized_sections:
                optimized_sections[tag] = content
        return optimized_sections
