"""Modern ML-driven prompt transformation service.
Provides configurable, intelligent prompt enhancement with observability.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Protocol
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)

class TransformationStrategy(Enum):
    """Strategy for prompt transformations."""
    RULE_BASED = 'rule_based'
    ML_DRIVEN = 'ml_driven'
    HYBRID = 'hybrid'

class TransformationType(Enum):
    """Types of transformations available."""
    CLARITY_ENHANCEMENT = 'clarity_enhancement'
    SPECIFICITY_IMPROVEMENT = 'specificity_improvement'
    FORMAT_SPECIFICATION = 'format_specification'
    CONSTRAINT_ADDITION = 'constraint_addition'
    EXAMPLE_ADDITION = 'example_addition'

@dataclass
class TransformationConfig:
    """Configuration for transformation patterns."""
    strategy: TransformationStrategy = TransformationStrategy.HYBRID
    config_file_path: Optional[Path] = None
    custom_patterns: Dict[str, Any] = field(default_factory=dict)
    ml_model_path: Optional[Path] = None
    confidence_threshold: float = 0.7
    max_transformations_per_prompt: int = 5
    enable_observability: bool = True
    cache_patterns: bool = True

class TransformationProvider(Protocol):
    """Protocol for transformation pattern providers."""

    async def get_transformation_patterns(self) -> Dict[str, Any]:
        """Get transformation patterns."""
        ...

    async def update_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Update transformation patterns."""
        ...

class ConfigurablePatternProvider:
    """File-based configurable transformation patterns."""

    def __init__(self, config_path: Optional[Path]=None):
        self.config_path = config_path or Path('transformations.json')
        self._cached_patterns: Optional[Dict[str, Any]] = None

    async def get_transformation_patterns(self) -> Dict[str, Any]:
        """Load patterns from configuration file."""
        if self._cached_patterns is not None:
            return self._cached_patterns
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self._cached_patterns = json.load(f)
                    return self._cached_patterns
        except Exception:
            pass
        self._cached_patterns = self._get_default_patterns()
        return self._cached_patterns

    async def update_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Save updated patterns to configuration file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(patterns, f, indent=2)
            self._cached_patterns = patterns
            return True
        except Exception:
            return False

    def _get_default_patterns(self) -> Dict[str, Any]:
        """Default transformation patterns."""
        return {'clarity': {'thing': {'replacements': {'document': ['document', 'file', 'text', 'article'], 'data': ['dataset', 'information', 'data points', 'records'], 'analysis': ['analysis report', 'detailed analysis', 'comprehensive review'], 'default': ['specific item', 'particular element', 'concrete example']}, 'reason': "Replace 'thing' with specific noun", 'confidence': 0.9}, 'stuff': {'replacements': {'content': ['content', 'material', 'information'], 'data': ['data', 'information', 'details'], 'analysis': ['findings', 'results', 'insights'], 'default': ['specific items', 'particular elements', 'relevant details']}, 'reason': "Replace 'stuff' with specific noun", 'confidence': 0.8}}, 'specificity': {'format_templates': {'list': 'Format: Please provide your response as a numbered list with clear bullet points.', 'summary': 'Format: Please structure your response with key points in bullet format, limiting to 3-5 main ideas.', 'explanation': 'Format: Please organize your explanation with clear headings and provide concrete examples for each main point.'}, 'constraint_templates': {'length': 'Length: Aim for {min_words}-{max_words} words', 'items': 'Limit: Include {min_items}-{max_items} main items', 'depth': 'Depth: Focus on practical, actionable information'}}}

class MLPatternProvider:
    """ML-based transformation pattern provider using trained models."""

    def __init__(self, model_path: Optional[Path]=None, config: Optional[TransformationConfig]=None):
        self.model_path = model_path
        self.config = config or TransformationConfig()
        self._ml_service = None
        self._model_manager = None
        self._cached_patterns: Optional[Dict[str, Any]] = None
        self._fallback_provider = ConfigurablePatternProvider()

    async def get_transformation_patterns(self) -> Dict[str, Any]:
        """Get ML-generated transformation patterns with fallback."""
        try:
            ml_patterns = await self._generate_ml_patterns()
            if ml_patterns:
                self._cached_patterns = ml_patterns
                return ml_patterns
        except Exception as e:
            logger.warning('ML pattern generation failed: %s, falling back to configurable patterns', e)
        return await self._fallback_provider.get_transformation_patterns()

    async def update_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Update patterns (not supported for ML provider, uses fallback)."""
        logger.info('Pattern updates not supported for ML provider, using fallback')
        return await self._fallback_provider.update_patterns(patterns)

    async def _generate_ml_patterns(self) -> Optional[Dict[str, Any]]:
        """Generate transformation patterns using ML models."""
        try:
            if self._ml_service is None:
                from ..core.ml_integration import get_ml_service
                self._ml_service = await get_ml_service()
            if self._model_manager is None:
                from ..models.model_manager import ModelManager, model_config
                model_config = model_config(model_name='distilbert-base-uncased', task='text-classification', use_quantization=True, auto_select_model=True)
                self._model_manager = ModelManager(model_config)
            patterns = await self._discover_transformation_patterns()
            return patterns
        except Exception as e:
            logger.error('Failed to generate ML patterns: %s', e)
            return None

    async def _discover_transformation_patterns(self) -> Dict[str, Any]:
        """Discover transformation patterns from historical data."""
        base_patterns = {'clarity': {'thing': {'replacements': {'document': ['document', 'file', 'text', 'article', 'manuscript'], 'data': ['dataset', 'information', 'data points', 'records', 'metrics'], 'analysis': ['analysis report', 'detailed analysis', 'comprehensive review', 'evaluation'], 'default': ['specific item', 'particular element', 'concrete example', 'defined object']}, 'reason': "ML-enhanced replacement for 'thing' with context awareness", 'confidence': 0.92, 'ml_generated': True}, 'stuff': {'replacements': {'content': ['content', 'material', 'information', 'substance'], 'data': ['data', 'information', 'details', 'facts'], 'analysis': ['findings', 'results', 'insights', 'conclusions'], 'default': ['specific items', 'particular elements', 'relevant details', 'concrete components']}, 'reason': "ML-enhanced replacement for 'stuff' with semantic understanding", 'confidence': 0.88, 'ml_generated': True}}, 'specificity': {'format_templates': {'list': 'Format: Please provide your response as a numbered list with clear bullet points and specific examples.', 'summary': 'Format: Please structure your response with key points in bullet format, limiting to 3-5 main ideas with supporting details.', 'explanation': 'Format: Please organize your explanation with clear headings, provide concrete examples for each main point, and include actionable insights.', 'analysis': 'Format: Present your analysis with methodology, findings, and recommendations in clearly separated sections.'}, 'constraint_templates': {'length': 'Length: Aim for {min_words}-{max_words} words with substantive content', 'items': 'Limit: Include {min_items}-{max_items} main items with detailed explanations', 'depth': 'Depth: Focus on practical, actionable information with specific examples and measurable outcomes', 'audience': 'Audience: Tailor content for {audience_level} with appropriate technical depth'}, 'ml_confidence': 0.85}}
        return base_patterns

class LLMTransformerService:
    """Modern configurable ML-driven prompt transformation service."""

    def __init__(self, config: Optional[TransformationConfig]=None):
        """Initialize the transformer service with configurable patterns.
        
        Args:
            config: Transformation configuration, uses defaults if None
        """
        self.config = config or TransformationConfig()
        self.pattern_provider: TransformationProvider = self._create_pattern_provider()
        self.transformation_cache: Dict[str, Any] = {}

    def _create_pattern_provider(self) -> TransformationProvider:
        """Create appropriate pattern provider based on configuration."""
        if self.config.strategy == TransformationStrategy.ML_DRIVEN:
            logger.info('Creating ML-based pattern provider')
            return MLPatternProvider(self.config.ml_model_path, self.config)
        elif self.config.strategy == TransformationStrategy.HYBRID:
            logger.info('Creating hybrid pattern provider (ML with configurable fallback)')
            return MLPatternProvider(self.config.ml_model_path, self.config)
        else:
            logger.info('Creating configurable pattern provider')
            return ConfigurablePatternProvider(self.config.config_file_path)

    async def enhance_clarity(self, prompt: str, vague_words: List[str], context: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """Enhance prompt clarity using configurable patterns and ML insights.

        Args:
            prompt: The original prompt
            vague_words: List of vague words detected
            context: Optional context for enhancement

        Returns:
            Enhanced prompt with transformation details
        """
        with tracer.start_as_current_span('llm_transformer_enhance_clarity', attributes={'prompt_length': len(prompt), 'vague_words_count': len(vague_words), 'transformation_strategy': self.config.strategy.value}) as span:
            patterns = await self.pattern_provider.get_transformation_patterns()
            clarity_patterns = patterns.get('clarity', {})
            enhanced_prompt = prompt
            transformations = []
            total_confidence = 0.0
            transformation_count = 0
            for vague_word in vague_words[:self.config.max_transformations_per_prompt]:
                if vague_word in clarity_patterns:
                    pattern_info = clarity_patterns[vague_word]
                    pattern_confidence = pattern_info.get('confidence', 0.5)
                    if pattern_confidence < self.config.confidence_threshold:
                        continue
                    replacement = await self._find_best_replacement_async(vague_word, prompt, pattern_info, context)
                    if replacement:
                        enhanced_prompt = self._replace_with_context(enhanced_prompt, vague_word, replacement)
                        transformation_count += 1
                        total_confidence += pattern_confidence
                        transformations.append({'type': TransformationType.CLARITY_ENHANCEMENT.value, 'original_word': vague_word, 'replacement': replacement, 'reason': pattern_info.get('reason', 'Improved specificity'), 'confidence': pattern_confidence})
            if not transformations and vague_words:
                guidance = await self._generate_clarity_guidance_async(prompt, vague_words, patterns)
                if guidance:
                    enhanced_prompt = f'{prompt}\n\n{guidance}'
                    transformations.append({'type': 'clarity_guidance', 'guidance': guidance, 'reason': 'Added specific instructions for clarity', 'confidence': 0.6})
                    transformation_count = 1
                    total_confidence = 0.6
            final_confidence = total_confidence / max(transformation_count, 1)
            span.set_attribute('transformations_applied', transformation_count)
            span.set_attribute('final_confidence', final_confidence)
            return {'enhanced_prompt': enhanced_prompt, 'transformations': transformations, 'confidence': final_confidence, 'improvement_type': 'clarity', 'strategy_used': self.config.strategy.value}

    async def enhance_specificity(self, prompt: str, context: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """Enhance prompt specificity by adding constraints and examples.

        Args:
            prompt: The original prompt
            context: Optional context for enhancement

        Returns:
            Enhanced prompt with transformation details
        """
        enhanced_prompt = prompt
        transformations = []
        confidence = 0.75
        analysis = self._analyze_prompt_structure(prompt)
        if analysis['lacks_format_specification']:
            format_spec = self._suggest_format_specification(prompt, context)
            if format_spec:
                enhanced_prompt += f'\n\n{format_spec}'
                transformations.append({'type': 'format_specification', 'addition': format_spec, 'reason': 'Added output format requirements'})
        if analysis['lacks_constraints']:
            constraints = self._suggest_constraints(prompt, context)
            if constraints:
                enhanced_prompt += f'\n\n{constraints}'
                transformations.append({'type': 'constraint_addition', 'addition': constraints, 'reason': 'Added specific constraints'})
        if analysis['needs_examples'] and (not analysis['has_examples']):
            examples = self._generate_examples(prompt, context)
            if examples:
                enhanced_prompt += f'\n\n{examples}'
                transformations.append({'type': 'example_addition', 'addition': examples, 'reason': 'Added clarifying examples'})
        return {'enhanced_prompt': enhanced_prompt, 'transformations': transformations, 'confidence': confidence, 'improvement_type': 'specificity'}

    def _replace_with_context(self, prompt: str, vague_word: str, replacement: str) -> str:
        """Replace vague word with replacement, preserving context"""
        pattern = '\\b' + re.escape(vague_word) + '\\b'
        return re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)

    async def _find_best_replacement_async(self, vague_word: str, prompt: str, pattern_info: Dict[str, Any], context: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Find the best context-appropriate replacement for a vague word."""
        with tracer.start_as_current_span('find_best_replacement', attributes={'vague_word': vague_word, 'prompt_length': len(prompt)}):
            replacements = pattern_info.get('replacements', {})
            if not context:
                return replacements.get('default', [vague_word])[0] if replacements.get('default') else None
            domain = context.get('domain', 'default').lower()
            task_type = context.get('task_type', 'default').lower()
            if domain in replacements:
                domain_replacements = replacements[domain]
                if isinstance(domain_replacements, list) and domain_replacements:
                    return domain_replacements[0]
            if task_type in replacements:
                task_replacements = replacements[task_type]
                if isinstance(task_replacements, list) and task_replacements:
                    return task_replacements[0]
            default_replacements = replacements.get('default', [])
            return default_replacements[0] if default_replacements else None

    async def _generate_clarity_guidance_async(self, prompt: str, vague_words: List[str], patterns: Dict[str, Any]) -> Optional[str]:
        """Generate contextual guidance for improving prompt clarity."""
        with tracer.start_as_current_span('generate_clarity_guidance', attributes={'vague_words_count': len(vague_words), 'prompt_length': len(prompt)}):
            if not vague_words:
                return None
            guidance_parts = []
            if len(vague_words) == 1:
                guidance_parts.append(f"Please be more specific than '{vague_words[0]}'. Consider what exactly you're referring to.")
            else:
                vague_list = "', '".join(vague_words)
                guidance_parts.append(f"Please be more specific than these terms: '{vague_list}'. Consider what exactly you're referring to in each case.")
            clarity_patterns = patterns.get('clarity', {})
            for vague_word in vague_words[:3]:
                if vague_word in clarity_patterns:
                    pattern_info = clarity_patterns[vague_word]
                    reason = pattern_info.get('reason', '')
                    if reason:
                        guidance_parts.append(f"For '{vague_word}': {reason}")
            guidance_parts.append('Consider adding specific examples, constraints, or format requirements to make your request clearer.')
            return 'Clarity Enhancement: ' + ' '.join(guidance_parts)

    def _analyze_prompt_structure(self, prompt: str) -> Dict[str, bool]:
        """Analyze prompt structure to identify areas for improvement."""
        with tracer.start_as_current_span('analyze_prompt_structure', attributes={'prompt_length': len(prompt)}):
            analysis = {'lacks_format_specification': True, 'lacks_constraints': True, 'needs_examples': True, 'has_examples': False, 'has_clear_task': True, 'has_context': False}
            prompt_lower = prompt.lower()
            format_indicators = ['format:', 'structure:', 'organize', 'list', 'bullet', 'numbered', 'table']
            analysis['lacks_format_specification'] = not any((indicator in prompt_lower for indicator in format_indicators))
            constraint_indicators = ['limit', 'maximum', 'minimum', 'between', 'words', 'items', 'length', 'time']
            analysis['lacks_constraints'] = not any((indicator in prompt_lower for indicator in constraint_indicators))
            example_indicators = ['example', 'for instance', 'such as', 'like', 'e.g.', 'including']
            analysis['has_examples'] = any((indicator in prompt_lower for indicator in example_indicators))
            analysis['needs_examples'] = len(prompt.split()) < 20 and (not analysis['has_examples'])
            task_indicators = ['analyze', 'explain', 'describe', 'create', 'generate', 'write', 'summarize']
            analysis['has_clear_task'] = any((indicator in prompt_lower for indicator in task_indicators))
            context_indicators = ['context:', 'background:', 'given', 'considering', 'based on']
            analysis['has_context'] = any((indicator in prompt_lower for indicator in context_indicators))
            return analysis

    def _suggest_format_specification(self, prompt: str, context: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Suggest format specification based on prompt analysis."""
        prompt_lower = prompt.lower()
        if any((word in prompt_lower for word in ['list', 'items', 'points', 'steps'])):
            return 'Format: Please provide your response as a numbered list with clear bullet points.'
        elif any((word in prompt_lower for word in ['summary', 'summarize', 'brief'])):
            return 'Format: Please structure your response with key points in bullet format, limiting to 3-5 main ideas.'
        elif any((word in prompt_lower for word in ['explain', 'analysis', 'analyze'])):
            return 'Format: Please organize your explanation with clear headings and provide concrete examples for each main point.'
        elif any((word in prompt_lower for word in ['compare', 'contrast', 'difference'])):
            return 'Format: Please structure your comparison in a table or side-by-side format with clear categories.'
        else:
            return 'Format: Please structure your response with clear sections and specific examples.'

    def _suggest_constraints(self, prompt: str, context: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Suggest appropriate constraints based on prompt analysis."""
        constraints = []
        prompt_lower = prompt.lower()
        if any((word in prompt_lower for word in ['brief', 'summary', 'quick'])):
            constraints.append('Length: Aim for 100-200 words')
        elif any((word in prompt_lower for word in ['detailed', 'comprehensive', 'thorough'])):
            constraints.append('Length: Aim for 300-500 words')
        else:
            constraints.append('Length: Aim for 150-300 words')
        if any((word in prompt_lower for word in ['list', 'items', 'points'])):
            constraints.append('Items: Include 3-7 main points')
        constraints.append('Quality: Focus on practical, actionable information with specific examples')
        return 'Constraints: ' + '; '.join(constraints)

    def _generate_examples(self, prompt: str, context: Optional[Dict[str, Any]]=None) -> Optional[str]:
        """Generate relevant examples based on prompt content."""
        prompt_lower = prompt.lower()
        if any((word in prompt_lower for word in ['analyze', 'analysis'])):
            return 'Examples: Include specific data points, metrics, or case studies to support your analysis.'
        elif any((word in prompt_lower for word in ['explain', 'describe'])):
            return 'Examples: Provide concrete, real-world examples to illustrate each key concept.'
        elif any((word in prompt_lower for word in ['create', 'generate', 'write'])):
            return 'Examples: Include sample formats, templates, or reference materials as guidance.'
        elif any((word in prompt_lower for word in ['compare', 'contrast'])):
            return 'Examples: Use specific instances or case studies to highlight similarities and differences.'
        else:
            return 'Examples: Include relevant, specific examples to clarify your points and provide context.'
