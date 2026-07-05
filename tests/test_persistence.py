import sqlalchemy
import sqlalchemy.exc
from sqlalchemy import inspect

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
