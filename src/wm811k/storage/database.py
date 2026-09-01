"""PostgreSQL connection helpers."""

from __future__ import annotations

import os

import psycopg
from psycopg import Connection

DATABASE_URL_ENV = "DATABASE_URL"

def get_database_url() -> str:
    """Read the PostgreSQL connection URL grom the environment."""

    database_url = os.getenv(DATABASE_URL_ENV)
    if not database_url:
        raise RuntimeError(
            f"{DATABASE_URL_ENV} is not set."
            "Set it to a PostgreSQL connection URL before running database commands."
        )
    return database_url

def connect_database() -> Connection:
    """Open a PostgreSQL database connection."""

    return psycopg.connect(get_database_url())

def check_database_connection() -> tuple[str, str]:
    """Return the connected database name and PostgreSQL user."""

    with connect_database() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            result = cursor.fetchone()

    if result is None:
        raise RuntimeError("PostgreSQL returned no connection information")

    return str(result[0]), str(result[1])