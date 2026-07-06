"""add trade quality score

Revision ID: d8f3b2a1c9e4
Revises: c4e9a1f2d3b7
Create Date: 2026-07-06 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd8f3b2a1c9e4'
down_revision: Union[str, Sequence[str], None] = 'c4e9a1f2d3b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('analysis_run', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('weighting_profile', sa.String(), nullable=False, server_default='swing')
        )

    with op.batch_alter_table('scored_candidate', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('open_interest', sa.Integer(), nullable=False, server_default='0')
        )
        batch_op.add_column(
            sa.Column('spread_pct', sa.Float(), nullable=False, server_default='0')
        )
        batch_op.add_column(
            sa.Column('volume', sa.Integer(), nullable=False, server_default='0')
        )
        batch_op.add_column(
            sa.Column('domain_scores', sa.JSON(), nullable=False, server_default='{}')
        )
        batch_op.drop_column('score_breakdown')

    op.create_table(
        'trade_quality_score',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('contract_symbol', sa.String(), nullable=True),
        sa.Column('composite_score', sa.Float(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('recommendation_action', sa.String(), nullable=False),
        sa.Column('weighting_profile', sa.String(), nullable=False),
        sa.Column('domain_scores', sa.JSON(), nullable=False),
        sa.Column('explainability', sa.JSON(), nullable=False),
        sa.Column('generated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['analysis_run.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id'),
    )

    with op.batch_alter_table('agent_thesis', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('quant_trade_quality', sa.JSON(), nullable=False, server_default='{}')
        )
        batch_op.add_column(
            sa.Column('technical_domain_score', sa.JSON(), nullable=False, server_default='{}')
        )
        batch_op.add_column(sa.Column('fundamental_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('sentiment_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('macro_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('relative_strength_narrative', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('relative_strength_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('statistical_edge_narrative', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('statistical_edge_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('risk_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('liquidity_domain_score', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('agent_trade_quality', sa.JSON(), nullable=True))
        batch_op.drop_column('quant_score_breakdown')
        batch_op.drop_column('quant_overall_score')


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('agent_thesis', schema=None) as batch_op:
        batch_op.add_column(sa.Column('quant_overall_score', sa.Float(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('quant_score_breakdown', sa.JSON(), nullable=False, server_default='{}'))
        batch_op.drop_column('agent_trade_quality')
        batch_op.drop_column('liquidity_domain_score')
        batch_op.drop_column('risk_domain_score')
        batch_op.drop_column('statistical_edge_domain_score')
        batch_op.drop_column('statistical_edge_narrative')
        batch_op.drop_column('relative_strength_domain_score')
        batch_op.drop_column('relative_strength_narrative')
        batch_op.drop_column('macro_domain_score')
        batch_op.drop_column('sentiment_domain_score')
        batch_op.drop_column('fundamental_domain_score')
        batch_op.drop_column('technical_domain_score')
        batch_op.drop_column('quant_trade_quality')

    op.drop_table('trade_quality_score')

    with op.batch_alter_table('scored_candidate', schema=None) as batch_op:
        batch_op.add_column(sa.Column('score_breakdown', sa.JSON(), nullable=False, server_default='{}'))
        batch_op.drop_column('domain_scores')
        batch_op.drop_column('volume')
        batch_op.drop_column('spread_pct')
        batch_op.drop_column('open_interest')

    with op.batch_alter_table('analysis_run', schema=None) as batch_op:
        batch_op.drop_column('weighting_profile')
