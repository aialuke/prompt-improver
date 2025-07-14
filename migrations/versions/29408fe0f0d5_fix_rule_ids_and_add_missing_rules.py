"""Fix rule IDs and add missing rules

Revision ID: 29408fe0f0d5
Revises: seed_001
Create Date: 2025-07-13 22:44:47.052077

"""
from typing import Sequence, Union
import json
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy import table, column, String, Integer, Boolean, DateTime, Text


# revision identifiers, used by Alembic.
revision: str = '29408fe0f0d5'
down_revision: Union[str, Sequence[str], None] = 'seed_001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fix rule IDs and add missing rules."""
    # Define table structure for operations
    rule_metadata = table('rule_metadata',
        column('rule_id', String),
        column('rule_name', String),
        column('rule_category', String),
        column('rule_description', Text),
        column('enabled', Boolean),
        column('priority', Integer),
        column('rule_version', String),
        column('default_parameters', Text),  # JSON string
        column('parameter_constraints', Text),  # JSON string
        column('created_at', DateTime),
        column('updated_at', DateTime)
    )
    
    # Delete existing incorrect rules
    op.execute("DELETE FROM rule_metadata WHERE rule_version = '1.0.0'")
    
    # Insert correct rules with proper IDs
    now = datetime.utcnow()
    correct_rules = [
        {
            'rule_id': 'clarity_enhancement',
            'rule_name': 'Clarity Enhancement Rule',
            'rule_category': 'fundamental',
            'rule_description': 'Improves prompt clarity using research-validated patterns from Anthropic and OpenAI documentation. Replaces vague requests with specific, measurable outcomes and applies XML structure patterns.',
            'enabled': True,
            'priority': 10,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'min_clarity_score': 0.7,
                'sentence_complexity_threshold': 20,
                'use_structured_xml': True,
                'apply_specificity_patterns': True,
                'add_success_criteria': True,
                'context_placement_priority': 'before_examples'
            }),
            'parameter_constraints': json.dumps({
                'min_clarity_score': {'min': 0.0, 'max': 1.0},
                'sentence_complexity_threshold': {'min': 10, 'max': 50},
                'use_structured_xml': {'type': 'boolean'},
                'apply_specificity_patterns': {'type': 'boolean'}
            }),
            'created_at': now,
            'updated_at': now
        },
        {
            'rule_id': 'specificity_enhancement',
            'rule_name': 'Specificity and Detail Rule',
            'rule_category': 'fundamental',
            'rule_description': 'Reduces vague language and increases prompt specificity using multi-source research patterns. Enforces measurable goals and specific success criteria.',
            'enabled': True,
            'priority': 9,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'vague_language_threshold': 0.3,
                'require_specific_outcomes': True,
                'include_success_criteria': True,
                'enforce_measurable_goals': True,
                'specificity_patterns': ['who_what_when_where', 'concrete_examples', 'quantifiable_metrics'],
                'avoid_hedge_words': True
            }),
            'parameter_constraints': json.dumps({
                'vague_language_threshold': {'min': 0.0, 'max': 1.0},
                'require_specific_outcomes': {'type': 'boolean'},
                'include_success_criteria': {'type': 'boolean'},
                'enforce_measurable_goals': {'type': 'boolean'}
            }),
            'created_at': now,
            'updated_at': now
        },
        {
            'rule_id': 'chain_of_thought',
            'rule_name': 'Chain of Thought Reasoning Rule',
            'rule_category': 'reasoning',
            'rule_description': 'Implements step-by-step reasoning patterns based on CoT research across multiple LLM providers. Uses zero-shot and few-shot CoT techniques with structured thinking tags.',
            'enabled': True,
            'priority': 8,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'enable_step_by_step': True,
                'use_thinking_tags': True,
                'min_reasoning_steps': 3,
                'encourage_explicit_reasoning': True,
                'zero_shot_trigger': 'Let\'s think step by step',
                'use_structured_response': True
            }),
            'parameter_constraints': json.dumps({
                'min_reasoning_steps': {'min': 1, 'max': 10},
                'enable_step_by_step': {'type': 'boolean'},
                'use_thinking_tags': {'type': 'boolean'}
            }),
            'created_at': now,
            'updated_at': now
        },
        {
            'rule_id': 'few_shot_examples',
            'rule_name': 'Few-Shot Example Integration Rule',
            'rule_category': 'examples',
            'rule_description': 'Incorporates 2-5 optimal examples based on research from PromptHub and OpenAI documentation. Uses diverse examples with XML delimiters and balanced positive/negative cases.',
            'enabled': True,
            'priority': 7,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'optimal_example_count': 3,
                'require_diverse_examples': True,
                'include_negative_examples': True,
                'use_xml_delimiters': True,
                'example_placement': 'after_context',
                'recency_bias_optimization': True
            }),
            'parameter_constraints': json.dumps({
                'optimal_example_count': {'min': 2, 'max': 5},
                'require_diverse_examples': {'type': 'boolean'},
                'include_negative_examples': {'type': 'boolean'}
            }),
            'created_at': now,
            'updated_at': now
        },
        {
            'rule_id': 'role_based_prompting',
            'rule_name': 'Expert Role Assignment Rule',
            'rule_category': 'context',
            'rule_description': 'Assigns appropriate expert personas based on Anthropic best practices for role-based prompting. Automatically detects domain and maintains persona consistency.',
            'enabled': True,
            'priority': 6,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'auto_detect_domain': True,
                'use_system_prompts': True,
                'maintain_persona_consistency': True,
                'expertise_depth': 'senior_level',
                'include_credentials': True
            }),
            'parameter_constraints': json.dumps({
                'auto_detect_domain': {'type': 'boolean'},
                'use_system_prompts': {'type': 'boolean'},
                'expertise_depth': {'values': ['junior', 'mid_level', 'senior_level', 'expert']}
            }),
            'created_at': now,
            'updated_at': now
        },
        {
            'rule_id': 'xml_structure_enhancement',
            'rule_name': 'XML Structure Enhancement Rule',
            'rule_category': 'structure',
            'rule_description': 'Implements XML tagging patterns recommended by Anthropic for Claude optimization. Uses context, instruction, example, thinking, and response tags for better organization.',
            'enabled': True,
            'priority': 5,
            'rule_version': '1.0.0',
            'default_parameters': json.dumps({
                'use_context_tags': True,
                'use_instruction_tags': True,
                'use_example_tags': True,
                'use_thinking_tags': True,
                'use_response_tags': True,
                'nested_structure_allowed': True,
                'attribute_usage': 'minimal'
            }),
            'parameter_constraints': json.dumps({
                'use_context_tags': {'type': 'boolean'},
                'use_instruction_tags': {'type': 'boolean'},
                'use_example_tags': {'type': 'boolean'},
                'use_thinking_tags': {'type': 'boolean'},
                'use_response_tags': {'type': 'boolean'}
            }),
            'created_at': now,
            'updated_at': now
        }
    ]
    
    # Bulk insert corrected rules
    op.bulk_insert(rule_metadata, correct_rules)
    
    print(f"✅ Fixed rule IDs and added {len(correct_rules)} research-validated rules")


def downgrade() -> None:
    """Downgrade - restore previous incorrect state."""
    # Delete the corrected rules
    op.execute("DELETE FROM rule_metadata WHERE rule_id IN ('clarity_enhancement', 'specificity_enhancement', 'chain_of_thought', 'few_shot_examples', 'role_based_prompting', 'xml_structure_enhancement')")
    
    # Note: We won't restore the old incorrect rules as they were buggy
    print("✅ Removed corrected rules - database is now empty for manual restoration if needed")
