"""Training data export routes."""

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from web.backend.services.persistence import get_persistence

router = APIRouter()


@router.get("/export")
async def export_training_data(
    batch_id: str | None = None,
    unexported_only: bool = True,
) -> JSONResponse:
    """Export trajectories in training-compatible format.

    Args:
        batch_id: Optional batch ID to export. If not provided, exports all unexported.
        unexported_only: If True (default), only export unexported trajectories.

    Returns:
        JSON response with trajectory count and data.
    """
    persistence = get_persistence()
    trajectories = persistence.export_training_data(
        batch_id=batch_id,
        unexported_only=unexported_only,
    )

    return JSONResponse(
        {
            "count": len(trajectories),
            "trajectories": trajectories,
        }
    )


@router.post("/mark-exported/{batch_id}")
async def mark_exported(batch_id: str) -> dict[str, str]:
    """Mark a batch of trajectories as exported.

    This should be called after successfully using the training data
    to prevent re-exporting the same data.

    Args:
        batch_id: The batch ID to mark as exported.

    Returns:
        Status message.
    """
    persistence = get_persistence()
    persistence.mark_exported(batch_id)
    return {"status": "marked"}


@router.get("/stats")
async def get_training_stats() -> dict[str, Any]:
    """Get statistics about collected training data.

    Returns:
        Statistics including total trajectories, games, and export status.
    """
    persistence = get_persistence()
    return persistence.get_training_stats()


@router.get("/batches")
async def list_batches() -> list[dict[str, Any]]:
    """List all trajectory batches.

    Returns:
        List of batch summaries with counts and export status.
    """
    persistence = get_persistence()

    # Get batch info from database
    with persistence._get_connection() as conn:
        rows = conn.execute("""
            SELECT
                batch_id,
                game_id,
                COUNT(*) as trajectory_count,
                MIN(created_at) as created_at,
                MAX(exported) as exported
            FROM trajectories
            GROUP BY batch_id, game_id
            ORDER BY created_at DESC
        """).fetchall()

    return [
        {
            "batch_id": row["batch_id"],
            "game_id": row["game_id"],
            "trajectory_count": row["trajectory_count"],
            "created_at": row["created_at"],
            "exported": bool(row["exported"]),
        }
        for row in rows
    ]
