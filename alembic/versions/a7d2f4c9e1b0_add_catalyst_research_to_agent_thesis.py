"""add catalyst_research to agent_thesis

Revision ID: a7d2f4c9e1b0
Revises: 59ed11edc7f3
Create Date: 2026-07-04 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7d2f4c9e1b0'
down_revision: Union[str, Sequence[str], None] = '59ed11edc7f3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('agent_thesis', schema=None) as batch_op:
        batch_op.add_column(sa.Column('catalyst_net_bias', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('catalyst_summary', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('catalyst_items', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('agent_thesis', schema=None) as batch_op:
        batch_op.drop_column('catalyst_items')
        batch_op.drop_column('catalyst_summary')
        batch_op.drop_column('catalyst_net_bias')
