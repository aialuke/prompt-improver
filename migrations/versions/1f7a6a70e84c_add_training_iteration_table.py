"""add_training_iteration_table

Revision ID: 1f7a6a70e84c
Revises: update_training_v2
Create Date: 2025-07-24 14:21:34.588259

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1f7a6a70e84c'
down_revision: Union[str, Sequence[str], None] = 'update_training_v2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create training_iterations table for detailed iteration tracking
    op.create_table(
        'training_iterations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('iteration', sa.Integer(), nullable=False),
        sa.Column('workflow_id', sa.String(), nullable=True),
        sa.Column('performance_metrics', sa.JSON(), nullable=False),
        sa.Column('rule_optimizations', sa.JSON(), nullable=False),
        sa.Column('discovered_patterns', sa.JSON(), nullable=False),
        sa.Column('synthetic_data_generated', sa.Integer(), nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=False),
        sa.Column('improvement_score', sa.Float(), nullable=False),
        sa.Column('checkpoint_data', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['training_sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'iteration', name='unique_session_iteration')
    )

    # Create indexes for performance
    op.create_index('idx_training_iterations_session', 'training_iterations', ['session_id'])
    op.create_index('idx_training_iterations_workflow', 'training_iterations', ['workflow_id'])
    op.create_index('idx_training_iterations_performance', 'training_iterations', ['performance_metrics'], postgresql_using='gin')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('idx_training_iterations_performance', table_name='training_iterations')
    op.drop_index('idx_training_iterations_workflow', table_name='training_iterations')
    op.drop_index('idx_training_iterations_session', table_name='training_iterations')

    # Drop table
    op.drop_table('training_iterations')
