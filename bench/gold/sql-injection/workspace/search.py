import sqlite3


def search_users(db: sqlite3.Connection, name: str) -> list:
    # Seeded vulnerability: untrusted `name` is concatenated into the SQL.
    query = f"SELECT id, name FROM users WHERE name LIKE '%{name}%'"
    return db.execute(query).fetchall()
