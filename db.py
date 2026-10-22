"""Save and read predictions in a SQLite database."""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_FILE = Path(__file__).parent / "claimwise.db"


def init_db():
    """Create the predictions table if it does not exist yet."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                bmi REAL,
                children INTEGER,
                smoker TEXT,
                region TEXT,
                probability REAL,
                high_cost INTEGER
            )
            """
        )


def save_prediction(customer, probability, high_cost):
    """Insert one prediction and return its new id."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (created_at, age, sex, bmi, children, smoker, region, probability, high_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                customer.age,
                customer.sex,
                customer.bmi,
                customer.children,
                customer.smoker,
                customer.region,
                probability,
                int(high_cost),
            ),
        )
        return cursor.lastrowid


def get_recent_predictions(limit=10):
    """Return the most recent predictions, newest first."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]