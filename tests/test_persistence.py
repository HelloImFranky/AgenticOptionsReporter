import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import inspect

from agentic_options_reporter.models.db import Base
from agentic_options_reporter.persistence import run_migrations


def test_run_migrations_repair_sqlite_schema_on_existing_table_error(monkeypatch, tmp_path):
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    calls = []

    def fake_upgrade(cfg, revision):
        calls.append(("upgrade", revision))
        raise sqlalchemy.exc.OperationalError(
            "table analysis_run already exists",
            None,
            None,
        )

    def fake_stamp(cfg, revision):
        calls.append(("stamp", revision))

    monkeypatch.setattr("alembic.command.upgrade", fake_upgrade)
    monkeypatch.setattr("alembic.command.stamp", fake_stamp)

    run_migrations(database_url)

    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        inspector = inspect(engine)
        assert inspector.has_table("analysis_run")
        assert inspector.has_table("support_resistance_level")
    finally:
        engine.dispose()

    assert calls == [("upgrade", "head"), ("stamp", "head")]


def test_run_migrations_backfills_columns_on_pre_alembic_database(tmp_path):
    """A database whose tables predate Alembic tracking (no alembic_version
    row) hits "table already exists" on the very first migration, which
    aborts the whole upgrade chain before any later ADD COLUMN migration
    (fundamentals/data_warnings, catalyst_research, pipeline_warnings) runs.
    Regression: the recovery path used to stamp the revision as fully
    applied anyway, silently leaving those columns missing forever — every
    later attempt to persist a run then failed with 'no such column'."""
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        # Build today's full schema, then physically strip it back down to
        # what a database predating the fundamentals migration looked like.
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text(
                "CREATE TABLE analysis_run_old AS "
                "SELECT id, symbol, generated_at, lookback_days, expiration FROM analysis_run"
            ))
            conn.execute(sqlalchemy.text("DROP TABLE analysis_run"))
            conn.execute(sqlalchemy.text("ALTER TABLE analysis_run_old RENAME TO analysis_run"))
    finally:
        engine.dispose()

    run_migrations(database_url)

    engine = sqlalchemy.create_engine(database_url, connect_args={"check_same_thread": False})
    try:
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("analysis_run")}
        assert {"fundamentals", "data_warnings"} <= columns
    finally:
        engine.dispose()
