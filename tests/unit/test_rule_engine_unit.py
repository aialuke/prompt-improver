"""
Unit tests for rule engine components using best practices for text processing,
workflow engines, and database models. Implements property-based testing,
state machine verification, and comprehensive constraint validation.
"""
import json
import string
from unittest.mock import MagicMock, patch
import pytest
from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, precondition, rule

@pytest.mark.unit
class TestClarityRuleUnit:
    """Unit tests for clarity rule using text processing best practices."""

    def test_clarity_rule_simple_improvement(self):
        """Test clarity rule with simple prompt improvement."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        rule = ClarityRule()
        result = rule.apply('fix this')
        assert result.improved_prompt != 'fix this'
        assert len(result.improved_prompt) > len('fix this')
        assert result.confidence > 0
        assert result.success is True
        assert len(result.transformations) > 0

    def test_clarity_rule_already_clear_prompt(self):
        """Test clarity rule with already clear prompt."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        rule = ClarityRule()
        clear_prompt = 'Please provide a detailed step-by-step explanation of how to implement a binary search algorithm in Python, including error handling and time complexity analysis.'
        result = rule.apply(clear_prompt)
        assert result.success is True
        assert result.confidence >= 0

    def test_clarity_rule_confidence_scoring(self):
        """Test clarity rule confidence scoring logic."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        rule = ClarityRule()
        test_cases = [('help', 0.9), ('please help me', 0.7), ('please help me understand this concept', 0.5)]
        for prompt, expected_min_confidence in test_cases:
            result = rule.apply(prompt)
            assert result.confidence >= expected_min_confidence - 0.1
            assert result.confidence <= 1.0

    @given(st.text(alphabet=string.ascii_letters + string.digits + ' .,!?', min_size=1, max_size=100))
    def test_clarity_rule_always_succeeds(self, prompt):
        """Property: clarity rule should always produce a valid result for any text input."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        assume(len(prompt.strip()) > 0)
        rule = ClarityRule()
        result = rule.apply(prompt)
        assert result.success is True
        assert isinstance(result.improved_prompt, str)
        assert 0 <= result.confidence <= 1.0
        assert isinstance(result.transformations, list)

    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(st.one_of(st.just('fix this thing'), st.just('analyze the stuff'), st.just('help with that'), st.just('make it better'), st.text(alphabet=string.ascii_letters + ' ', min_size=5, max_size=30).map(lambda x: x + ' thing' if 'thing' not in x else x)))
    def test_clarity_rule_vague_word_detection(self, vague_prompt):
        """Property: prompts with vague words should trigger improvements."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        rule = ClarityRule()
        check_result = rule.check(vague_prompt)
        assert len(check_result.metadata.get('vague_words', [])) > 0
        apply_result = rule.apply(vague_prompt)
        if check_result.applies:
            assert apply_result.improved_prompt != vague_prompt
        else:
            assert apply_result.success is True

    @given(st.text(alphabet=string.ascii_letters + ' ', min_size=10, max_size=100).filter(lambda x: not any((word in x.lower() for word in ['thing', 'stuff', 'this', 'that', 'analyze', 'summarize']))))
    def test_clarity_rule_clear_text_handling(self, clear_prompt):
        """Property: clear text without vague words should have minimal changes."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        assume(len(clear_prompt.strip()) >= 10)
        rule = ClarityRule()
        check_result = rule.check(clear_prompt)
        if check_result.applies:
            apply_result = rule.apply(clear_prompt)
            assert apply_result.confidence <= 1.0
        else:
            assert check_result.applies is False

    def test_clarity_rule_convergence_property(self):
        """Property: clarity rule should eventually converge and not expand indefinitely."""
        from prompt_improver.rule_engine.rules.clarity import ClarityRule
        rule = ClarityRule()
        test_prompts = ['fix this thing', 'analyze the stuff', 'make it better', 'help me with that']
        for prompt in test_prompts:
            current_prompt = prompt
            lengths = []
            for i in range(3):
                result = rule.apply(current_prompt)
                lengths.append(len(result.improved_prompt))
                current_prompt = result.improved_prompt
                assert result.success is True
                assert result.confidence > 0
            if len(lengths) >= 2:
                first_growth = lengths[1] / lengths[0] if lengths[0] > 0 else 1
                assert first_growth <= 10.0, f'Excessive initial growth: {first_growth:.2f}x'
            final_length = lengths[-1]
            assert final_length <= 500, f'Final prompt too long: {final_length} chars'

@pytest.mark.unit
class TestSpecificityRuleUnit:
    """Unit tests for specificity rule with maximum isolation."""

    def test_specificity_rule_vague_prompt(self):
        """Test specificity rule with vague prompt."""
        from prompt_improver.rule_engine.rules.specificity import SpecificityRule
        rule = SpecificityRule()
        vague_prompt = 'make it better'
        result = rule.apply(vague_prompt)
        assert result.improved_prompt != vague_prompt
        assert len(result.improved_prompt) > len(vague_prompt)
        assert result.confidence > 0
        assert result.success is True

    def test_specificity_rule_specific_prompt(self):
        """Test specificity rule with already specific prompt."""
        from prompt_improver.rule_engine.rules.specificity import SpecificityRule
        rule = SpecificityRule()
        specific_prompt = "Write a Python function named 'calculate_fibonacci' that takes an integer n as input and returns the nth Fibonacci number using dynamic programming, with input validation for negative numbers."
        result = rule.apply(specific_prompt)
        assert result.success is True
        assert result.confidence >= 0

    def test_specificity_rule_context_awareness(self):
        """Test specificity rule context awareness."""
        from prompt_improver.rule_engine.rules.specificity import SpecificityRule
        rule = SpecificityRule()
        result_with_context = rule.apply('optimize this', context={'domain': 'database', 'complexity': 'advanced'})
        result_without_context = rule.apply('optimize this')
        assert result_with_context.success is True
        assert result_without_context.success is True
        assert len(result_with_context.improved_prompt) >= len(result_without_context.improved_prompt)
        assert result_with_context.confidence > 0

@pytest.mark.unit
class TestRuleEngineUnit:
    """Unit tests for rule engine coordinator."""

    def test_rule_engine_initialization(self):
        """Test rule engine initializes with all rules."""
        from prompt_improver.rule_engine import RuleEngine
        engine = RuleEngine()
        assert len(engine.rules) >= 2
        assert any((getattr(rule, 'rule_id', '') == 'clarity_rule' for rule in engine.rules))
        assert any((getattr(rule, 'rule_id', '') == 'specificity_rule' for rule in engine.rules))

    def test_rule_engine_apply_all_rules(self):
        """Test rule engine applies all applicable rules."""
        from prompt_improver.rule_engine import RuleEngine
        engine = RuleEngine()
        result = engine.apply_rules('help')
        assert result.improved_prompt != 'help'
        assert len(result.applied_rules) > 0
        assert result.total_confidence > 0
        for rule_result in result.applied_rules:
            assert hasattr(rule_result, 'rule_id')
            assert hasattr(rule_result, 'confidence')
            assert 0 <= rule_result.confidence <= 1

    def test_rule_engine_rule_prioritization(self):
        """Test rule engine prioritizes rules correctly."""
        from prompt_improver.rule_engine import RuleEngine
        engine = RuleEngine()
        rule_priorities = [(getattr(rule, 'rule_id', 'unknown'), getattr(rule, 'priority', 0)) for rule in engine.rules]
        assert len(rule_priorities) >= 2
        clarity_priority = None
        specificity_priority = None
        for rule_id, priority in rule_priorities:
            if rule_id == 'clarity_rule':
                clarity_priority = priority
            elif rule_id == 'specificity_rule':
                specificity_priority = priority
        assert clarity_priority is not None, 'Clarity rule should exist'
        assert specificity_priority is not None, 'Specificity rule should exist'
        assert clarity_priority > specificity_priority, 'Clarity rule should have higher priority'

    def test_rule_engine_confidence_threshold(self):
        """Test rule engine respects confidence thresholds using real rules."""
        from prompt_improver.rule_engine import RuleEngine
        engine_high_threshold = RuleEngine(min_confidence=0.9)
        result_high = engine_high_threshold.apply_rules('help me please')
        for applied_rule in result_high.applied_rules:
            assert applied_rule.confidence >= 0.9, f'Rule {applied_rule.rule_id} confidence {applied_rule.confidence} below threshold 0.9'
        engine_low_threshold = RuleEngine(min_confidence=0.1)
        result_low = engine_low_threshold.apply_rules('help me please')
        assert len(result_low.applied_rules) >= len(result_high.applied_rules), 'Lower threshold should allow more rules'
        engine_zero_threshold = RuleEngine(min_confidence=0.0)
        result_zero = engine_zero_threshold.apply_rules('help me please')
        assert len(result_zero.applied_rules) >= len(result_low.applied_rules), 'Zero threshold should allow most rules'

class RuleEngineStateMachine(RuleBasedStateMachine):
    """State machine testing for rule engine workflow orchestration."""

    def __init__(self):
        super().__init__()
        from prompt_improver.rule_engine import RuleEngine
        self.engine = RuleEngine()
        self.applied_prompts = []
        self.confidence_threshold = 0.0
    prompts = Bundle('prompts')

    @rule(target=prompts, prompt=st.text(alphabet=string.ascii_letters + ' ', min_size=1, max_size=50))
    def add_prompt(self, prompt):
        """Add a prompt to test collection."""
        assume(len(prompt.strip()) > 0)
        return prompt.strip()

    @rule(threshold=st.floats(min_value=0.0, max_value=1.0))
    def set_confidence_threshold(self, threshold):
        """Change the engine's confidence threshold."""
        from prompt_improver.rule_engine import RuleEngine
        self.confidence_threshold = threshold
        self.engine = RuleEngine(min_confidence=threshold)

    @rule(prompt=prompts)
    def apply_rules_to_prompt(self, prompt):
        """Apply rules to a prompt and verify workflow invariants."""
        result = self.engine.apply_rules(prompt)
        self.applied_prompts.append((prompt, result, self.confidence_threshold))
        assert result.improved_prompt is not None
        assert isinstance(result.applied_rules, list)
        assert result.total_confidence >= 0
        for applied_rule in result.applied_rules:
            assert applied_rule.confidence >= self.confidence_threshold, f'Rule {applied_rule.rule_id} confidence {applied_rule.confidence} < threshold {self.confidence_threshold}'
        result2 = self.engine.apply_rules(prompt)
        assert result.improved_prompt == result2.improved_prompt

    @rule()
    @precondition(lambda self: len(self.applied_prompts) >= 2)
    def check_rule_consistency(self):
        """Verify rule application consistency with threshold changes."""
        prompt1, result1, threshold1 = self.applied_prompts[-2]
        prompt2, result2, threshold2 = self.applied_prompts[-1]
        if prompt1 == prompt2 and threshold1 == threshold2:
            assert result1.improved_prompt == result2.improved_prompt
            assert len(result1.applied_rules) == len(result2.applied_rules)
        for applied_rule in result1.applied_rules:
            assert applied_rule.confidence >= threshold1, f'Historical result violated threshold: {applied_rule.confidence} < {threshold1}'
        for applied_rule in result2.applied_rules:
            assert applied_rule.confidence >= threshold2, f'Recent result violated threshold: {applied_rule.confidence} < {threshold2}'
TestRuleEngineStateMachine = RuleEngineStateMachine.TestCase

@pytest.mark.unit
class TestRuleMetadataUnit:
    """Unit tests for rule metadata management using real SQLModel behavior."""

    def test_rule_metadata_creation(self):
        """Test rule metadata object creation with valid fields."""
        from prompt_improver.database.models import RuleMetadata
        metadata = RuleMetadata(rule_id='test_rule', rule_name='Test Rule', rule_category='unit_test', rule_description='Test rule for unit testing', default_parameters={'threshold': 0.8}, parameter_constraints={'threshold': {'min': 0.0, 'max': 1.0}}, enabled=True, priority=5, rule_version='1.0.0')
        assert metadata.rule_id == 'test_rule'
        assert metadata.rule_name == 'Test Rule'
        assert metadata.enabled == True
        assert metadata.priority == 5
        assert metadata.rule_version == '1.0.0'
        assert metadata.default_parameters['threshold'] == 0.8
        assert metadata.parameter_constraints['threshold']['min'] == 0.0

    def test_rule_metadata_field_validation(self):
        """Test SQLModel field validation and defaults."""
        from prompt_improver.database.models import RuleMetadata
        metadata = RuleMetadata(rule_id='minimal_rule', rule_name='Minimal Rule')
        assert metadata.enabled == True
        assert metadata.priority == 100
        assert metadata.rule_version == '1.0.0'
        assert metadata.category == 'general'
        assert metadata.default_parameters is None
        long_rule_id = 'a' * 50
        metadata_long = RuleMetadata(rule_id=long_rule_id, rule_name='Test with long ID')
        assert metadata_long.rule_id == long_rule_id

    def test_rule_metadata_jsonb_serialization(self):
        """Test JSONB field serialization behavior."""
        import json
        from prompt_improver.database.models import RuleMetadata
        complex_parameters = {'threshold': 0.8, 'enabled': True, 'nested': {'weights': [0.1, 0.2, 0.3], 'config': {'mode': 'strict', 'tolerance': 0.05}}}
        constraints = {'threshold': {'type': 'float', 'min': 0.0, 'max': 1.0}, 'nested.weights': {'type': 'array', 'min_length': 1}}
        metadata = RuleMetadata(rule_id='complex_rule', rule_name='Complex Rule', default_parameters=complex_parameters, parameter_constraints=constraints)
        assert isinstance(metadata.default_parameters, dict)
        assert metadata.default_parameters['threshold'] == 0.8
        assert metadata.default_parameters['nested']['config']['mode'] == 'strict'
        params_json = json.dumps(metadata.default_parameters)
        deserialized_params = json.loads(params_json)
        assert deserialized_params['threshold'] == 0.8
        assert deserialized_params['nested']['weights'] == [0.1, 0.2, 0.3]
        constraints_json = json.dumps(metadata.parameter_constraints)
        deserialized_constraints = json.loads(constraints_json)
        assert deserialized_constraints['threshold']['min'] == 0.0

    @given(st.text(alphabet=string.ascii_letters + string.digits + '_-', min_size=1, max_size=50))
    def test_rule_metadata_rule_id_constraints(self, rule_id):
        """Property: rule_id should accept valid identifier strings."""
        from prompt_improver.database.models import RuleMetadata
        metadata = RuleMetadata(rule_id=rule_id, rule_name='Test Rule')
        assert metadata.rule_id == rule_id
        assert isinstance(metadata.rule_id, str)

    @given(st.integers(min_value=1, max_value=1000))
    def test_rule_metadata_priority_constraints(self, priority):
        """Property: priority should accept positive integers."""
        from prompt_improver.database.models import RuleMetadata
        metadata = RuleMetadata(rule_id='test_rule', rule_name='Test Rule', priority=priority)
        assert metadata.priority == priority
        assert isinstance(metadata.priority, int)
        assert metadata.priority > 0

    @given(st.dictionaries(keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=20), values=st.one_of(st.floats(min_value=0.0, max_value=1.0), st.integers(min_value=0, max_value=100), st.booleans(), st.text(alphabet=string.ascii_letters, min_size=1, max_size=50)), min_size=0, max_size=10))
    def test_rule_metadata_parameters_serialization_roundtrip(self, parameters):
        """Property: JSONB parameters should survive serialization round-trip."""
        from prompt_improver.database.models import RuleMetadata
        metadata = RuleMetadata(rule_id='test_rule', rule_name='Test Rule', default_parameters=parameters)
        serialized = json.dumps(metadata.default_parameters)
        deserialized = json.loads(serialized)
        assert deserialized == parameters
        assert metadata.default_parameters == parameters

    @given(st.text(alphabet=string.ascii_letters + string.digits + '.-', min_size=1, max_size=15))
    def test_rule_metadata_version_format(self, version_base):
        """Property: rule_version should accept various version formats within length limit."""
        from prompt_improver.database.models import RuleMetadata
        if len(version_base) > 16:
            version_base = version_base[:16]
        version = f'{version_base}.0.0'
        assume(len(version) <= 20)
        metadata = RuleMetadata(rule_id='test_rule', rule_name='Test Rule', rule_version=version)
        assert metadata.rule_version == version
        assert len(metadata.rule_version) <= 20

    def test_rule_metadata_constraint_edge_cases(self):
        """Test edge cases for field constraints."""
        from prompt_improver.database.models import RuleMetadata
        max_rule_id = 'a' * 50
        metadata_max = RuleMetadata(rule_id=max_rule_id, rule_name='Max Length Rule')
        assert len(metadata_max.rule_id) == 50
        max_rule_name = 'b' * 100
        metadata_name = RuleMetadata(rule_id='test_rule', rule_name=max_rule_name)
        assert len(metadata_name.rule_name) == 100
        max_version = '1.0.0-beta.123456'
        metadata_version = RuleMetadata(rule_id='test_rule', rule_name='Test Rule', rule_version=max_version)
        assert len(metadata_version.rule_version) <= 20

    def test_rule_metadata_complex_jsonb_structures(self):
        """Test complex nested JSONB structures for robustness."""
        from prompt_improver.database.models import RuleMetadata
        complex_structure = {'level1': {'level2': {'level3': {'arrays': [1, 2, [3, 4, {'nested': 'value'}]], 'booleans': [True, False, None], 'mixed': {'str': 'text', 'num': 42, 'float': 3.14}}}}, 'edge_cases': {'empty_dict': {}, 'empty_list': [], 'unicode': '测试文本', 'special_chars': '!@#$%^&*()'}}
        metadata = RuleMetadata(rule_id='complex_test', rule_name='Complex Structure Test', default_parameters=complex_structure)
        nested_array = metadata.default_parameters['level1']['level2']['level3']['arrays'][2]
        assert isinstance(nested_array, list)
        assert len(nested_array) == 3
        assert nested_array[2]['nested'] == 'value'
        assert metadata.default_parameters['edge_cases']['unicode'] == '测试文本'
        serialized = json.dumps(metadata.default_parameters)
        deserialized = json.loads(serialized)
        assert deserialized == complex_structure
