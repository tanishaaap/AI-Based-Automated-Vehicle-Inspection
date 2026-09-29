"""
auth.py -- Lightweight local authentication + inspection history, backed by
SQLite (a single local .db file, no external server needed).

SECURITY NOTE: passwords are hashed with bcrypt before storage -- never
compare or store plaintext passwords. This replaces the placeholder
`if username and password:` check, which accepted literally anything.
"""

import sqlite3
import os
import json
import bcrypt
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "outputs", "app.db")


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image_name TEXT,
            vehicle_json TEXT,
            inspection_json TEXT,
            summary TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip()
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    conn = _get_connection()
    try:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "That username is already taken."
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        return True, "Account created successfully. Please log in."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> bool:
    if not username or not password:
        return False
    conn = _get_connection()
    try:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            return False
        return bcrypt.checkpw(password.encode("utf-8"), row[0])
    finally:
        conn.close()


def save_inspection(username: str, image_name: str, vehicle_info: dict, inspection_json: dict, summary: str):
    conn = _get_connection()
    try:
        conn.execute(
            """INSERT INTO inspections
               (username, image_name, vehicle_json, inspection_json, summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, image_name, json.dumps(vehicle_info), json.dumps(inspection_json),
             summary, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
    finally:
        conn.close()


def get_history(username: str, limit: int = 20) -> list[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            """SELECT image_name, vehicle_json, inspection_json, summary, created_at
               FROM inspections WHERE username = ? ORDER BY created_at DESC LIMIT ?""",
            (username, limit)
        ).fetchall()
        return [
            {
                "image_name": r[0],
                "vehicle": json.loads(r[1]),
                "inspection": json.loads(r[2]),
                "summary": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
    finally:
        conn.close()
