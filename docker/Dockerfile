FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* /app/
RUN poetry install --no-root --only main

COPY src /app/src
COPY specs /app/specs
COPY docs /app/docs
RUN poetry install --only-root

EXPOSE 8000

CMD ["uvicorn", "agentic_options_reporter.main:app", "--host", "0.0.0.0", "--port", "8000"]
