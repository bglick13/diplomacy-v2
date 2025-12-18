"""Persistence layer for game state and training data collection."""

import json
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PersistenceLayer(ABC):
    """Abstract persistence interface."""

    @abstractmethod
    def save_game(self, session_id: str, state: dict) -> None:
        """Save game state."""
        pass

    @abstractmethod
    def load_game(self, session_id: str) -> dict | None:
        """Load game state."""
        pass

    @abstractmethod
    def list_games(self, user_id: str | None = None) -> list[dict]:
        """List all games, optionally filtered by user."""
        pass

    @abstractmethod
    def delete_game(self, session_id: str) -> bool:
        """Delete a game."""
        pass

    @abstractmethod
    def save_trajectories(self, trajectories: list[dict], game_id: str) -> str:
        """Save trajectories and return batch_id."""
        pass

    @abstractmethod
    def export_training_data(
        self,
        batch_id: str | None = None,
        unexported_only: bool = True,
    ) -> list[dict]:
        """Export trajectories in training-compatible format."""
        pass

    @abstractmethod
    def mark_exported(self, batch_id: str) -> None:
        """Mark trajectories as exported."""
        pass

    @abstractmethod
    def get_training_stats(self) -> dict[str, Any]:
        """Get statistics about collected training data."""
        pass


class SQLitePersistence(PersistenceLayer):
    """SQLite-based persistence for MVP."""

    def __init__(self, db_path: str = "web/data/diplomacy.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    human_power TEXT NOT NULL,
                    adapter_name TEXT,
                    is_done BOOLEAN DEFAULT FALSE,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    user_id TEXT  -- For future auth
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    exported BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (game_id) REFERENCES games(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_batch
                ON trajectories(batch_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_game
                ON trajectories(game_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trajectories_exported
                ON trajectories(exported)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_games_user
                ON games(user_id)
            """)

    def save_game(self, session_id: str, state: dict) -> None:
        """Save or update game state."""
        now = time.time()
        with self._get_connection() as conn:
            # Check if game exists
            existing = conn.execute(
                "SELECT created_at FROM games WHERE id = ?", (session_id,)
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE games
                    SET state = ?, updated_at = ?, is_done = ?
                    WHERE id = ?
                """,
                    (
                        json.dumps(state),
                        now,
                        state.get("is_done", False),
                        session_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO games (id, state, human_power, adapter_name, is_done, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_id,
                        json.dumps(state),
                        state.get("human_power", "FRANCE"),
                        state.get("adapter_name"),
                        state.get("is_done", False),
                        now,
                        now,
                    ),
                )

    def load_game(self, session_id: str) -> dict | None:
        """Load game state."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT state FROM games WHERE id = ?", (session_id,)).fetchone()
            return json.loads(row["state"]) if row else None

    def list_games(self, user_id: str | None = None) -> list[dict]:
        """List all games, optionally filtered by user."""
        with self._get_connection() as conn:
            if user_id:
                rows = conn.execute(
                    """
                    SELECT id, human_power, adapter_name, is_done, created_at, updated_at
                    FROM games WHERE user_id = ?
                    ORDER BY updated_at DESC
                """,
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, human_power, adapter_name, is_done, created_at, updated_at
                    FROM games
                    ORDER BY updated_at DESC
                """).fetchall()

            return [dict(row) for row in rows]

    def delete_game(self, session_id: str) -> bool:
        """Delete a game and its trajectories."""
        with self._get_connection() as conn:
            # Delete trajectories first (foreign key)
            conn.execute("DELETE FROM trajectories WHERE game_id = ?", (session_id,))
            result = conn.execute("DELETE FROM games WHERE id = ?", (session_id,))
            return result.rowcount > 0

    def save_trajectories(self, trajectories: list[dict], game_id: str) -> str:
        """Save trajectories and return batch_id."""
        batch_id = f"web_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        now = time.time()

        with self._get_connection() as conn:
            for traj in trajectories:
                conn.execute(
                    """
                    INSERT INTO trajectories (batch_id, game_id, data, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (batch_id, game_id, json.dumps(traj), now),
                )

        return batch_id

    def export_training_data(
        self,
        batch_id: str | None = None,
        unexported_only: bool = True,
    ) -> list[dict]:
        """Export trajectories in training-compatible format."""
        with self._get_connection() as conn:
            if batch_id:
                rows = conn.execute(
                    "SELECT data FROM trajectories WHERE batch_id = ?",
                    (batch_id,),
                ).fetchall()
            elif unexported_only:
                rows = conn.execute(
                    "SELECT data FROM trajectories WHERE exported = FALSE"
                ).fetchall()
            else:
                rows = conn.execute("SELECT data FROM trajectories").fetchall()

            return [json.loads(row["data"]) for row in rows]

    def mark_exported(self, batch_id: str) -> None:
        """Mark trajectories as exported."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE trajectories SET exported = TRUE WHERE batch_id = ?",
                (batch_id,),
            )

    def get_training_stats(self) -> dict[str, Any]:
        """Get statistics about collected training data."""
        with self._get_connection() as conn:
            total_trajectories = conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]

            exported_trajectories = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE exported = TRUE"
            ).fetchone()[0]

            total_games = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

            completed_games = conn.execute(
                "SELECT COUNT(*) FROM games WHERE is_done = TRUE"
            ).fetchone()[0]

            # Get batch counts
            batch_counts = conn.execute("""
                SELECT batch_id, COUNT(*) as count
                FROM trajectories
                GROUP BY batch_id
            """).fetchall()

            return {
                "total_trajectories": total_trajectories,
                "exported_trajectories": exported_trajectories,
                "unexported_trajectories": total_trajectories - exported_trajectories,
                "total_games": total_games,
                "completed_games": completed_games,
                "batch_count": len(batch_counts),
                "avg_trajectories_per_batch": (
                    total_trajectories / len(batch_counts) if batch_counts else 0
                ),
            }


# Singleton instance
_persistence: PersistenceLayer | None = None


def get_persistence() -> PersistenceLayer:
    """Get the persistence layer singleton."""
    global _persistence
    if _persistence is None:
        _persistence = SQLitePersistence()
    return _persistence


def reset_persistence():
    """Reset the persistence singleton (for testing)."""
    global _persistence
    _persistence = None
