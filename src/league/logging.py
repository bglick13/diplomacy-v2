"""
WandB logging utilities for league training.

Provides visualization for:
- Elo progression over training
- League composition (baselines vs checkpoints)
- Win rates against gatekeepers
"""

from typing import Any

import wandb


def log_elo_update(
    step: int,
    challenger_name: str,
    challenger_elo: float,
    win_rate: float,
    games_played: int,
    all_elos: dict[str, float] | None = None,
) -> None:
    """
    Log Elo update to WandB.

    Args:
        step: Training step when evaluation occurred
        challenger_name: Name of the evaluated checkpoint
        challenger_elo: New Elo rating for challenger
        win_rate: Win rate against gatekeepers
        games_played: Number of games played
        all_elos: Optional dict of all agent Elos for league chart
    """
    log_data = {
        "elo/step": step,
        "elo/challenger": challenger_elo,
        "elo/win_rate": win_rate,
        "elo/games_played": games_played,
    }

    # Log individual agent Elos
    if all_elos:
        for agent_name, elo in all_elos.items():
            # Sanitize name for WandB (replace / with _)
            safe_name = agent_name.replace("/", "_")
            log_data[f"elo/agents/{safe_name}"] = elo

    wandb.log(log_data)


def log_league_summary(
    step: int,
    num_checkpoints: int,
    best_elo: float,
    best_agent: str,
    total_matches: int,
) -> None:
    """
    Log league summary statistics.

    Args:
        step: Current training step
        num_checkpoints: Total checkpoints in league
        best_elo: Highest Elo in the league
        best_agent: Name of agent with highest Elo
        total_matches: Total matches played for Elo computation
    """
    wandb.log(
        {
            "league/step": step,
            "league/num_checkpoints": num_checkpoints,
            "league/best_elo": best_elo,
            "league/total_matches": total_matches,
        }
    )


def create_elo_table(agents: list[dict[str, Any]]) -> wandb.Table:
    """
    Create a WandB table for Elo rankings.

    Args:
        agents: List of agent dicts with keys: name, type, elo, matches, step

    Returns:
        WandB Table for visualization
    """
    columns = ["Rank", "Agent", "Type", "Elo", "Matches", "Step"]
    data = []

    # Sort by Elo descending
    sorted_agents = sorted(agents, key=lambda x: x.get("elo", 0), reverse=True)

    for rank, agent in enumerate(sorted_agents, 1):
        data.append(
            [
                rank,
                agent.get("name", "unknown"),
                agent.get("type", "checkpoint"),
                agent.get("elo", 1000),
                agent.get("matches", 0),
                agent.get("step", 0),
            ]
        )

    return wandb.Table(columns=columns, data=data)


def log_elo_rankings(agents: list[dict[str, Any]], step: int) -> None:
    """
    Log Elo rankings table to WandB.

    Args:
        agents: List of agent info dicts
        step: Current training step
    """
    table = create_elo_table(agents)
    wandb.log(
        {
            "league/rankings": table,
            "league/rankings_step": step,
        }
    )


def create_matchup_heatmap(
    matchup_results: list[dict[str, Any]],
) -> wandb.Table:
    """
    Create a WandB table for matchup results (for heatmap visualization).

    Args:
        matchup_results: List of dicts with: challenger, opponent, win_rate, games

    Returns:
        WandB Table for heatmap
    """
    columns = ["Challenger", "Opponent", "Win Rate", "Games"]
    data = []

    for result in matchup_results:
        data.append(
            [
                result.get("challenger", "unknown"),
                result.get("opponent", "unknown"),
                result.get("win_rate", 0.0),
                result.get("games", 0),
            ]
        )

    return wandb.Table(columns=columns, data=data)
