"""Persist trained model artifacts (keras models, pickled scalers) to Neon
instead of relying on local disk, which is wiped on every Render redeploy/restart.
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv('DB_URI')


def _connect():
    return psycopg2.connect(DB_URI)


def _ensure_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_artifacts (
            model_name TEXT PRIMARY KEY,
            data BYTEA NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def save_artifact(model_name, file_path):
    """Upload a local file's bytes into Neon under model_name."""
    with open(file_path, 'rb') as f:
        data = f.read()

    conn = _connect()
    try:
        cur = conn.cursor()
        _ensure_table(cur)
        cur.execute("""
            INSERT INTO model_artifacts (model_name, data, updated_at)
            VALUES (%s, %s, now())
            ON CONFLICT (model_name) DO UPDATE SET data = EXCLUDED.data, updated_at = now()
        """, (model_name, psycopg2.Binary(data)))
        conn.commit()
        print(f"Saved artifact '{model_name}' ({len(data)} bytes) to Neon.")
    finally:
        conn.close()


def load_artifact(model_name, dest_path):
    """Download model_name's bytes from Neon and write them to dest_path.
    Returns True if an artifact was found and written, False otherwise.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        _ensure_table(cur)
        cur.execute("SELECT data FROM model_artifacts WHERE model_name = %s", (model_name,))
        row = cur.fetchone()
        if not row:
            return False
        with open(dest_path, 'wb') as f:
            f.write(bytes(row[0]))
        print(f"Restored artifact '{model_name}' from Neon -> {dest_path}")
        return True
    finally:
        conn.close()
