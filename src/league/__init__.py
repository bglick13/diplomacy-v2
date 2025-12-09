"""
League Training Module for Diplomacy GRPO.

This module provides infrastructure for:
- Agent registry (baselines + checkpoints)
- Multi-player Elo computation
- Opponent sampling (PFSP)
- Match history tracking

Usage:
    from src.league import LeagueRegistry, should_add_to_league
    from src.league.elo import update_elo_multiplayer
    from src.league.types import AgentInfo, OpponentType
"""

from src.league.elo import (
    batch_update_elo,
    calculate_expected_score,
    elo_diff_for_win_rate,
    estimate_win_probability,
    update_elo_from_match,
    update_elo_multiplayer,
)
from src.league.logging import (
    create_elo_table,
    create_matchup_heatmap,
    log_elo_rankings,
    log_elo_update,
    log_league_summary,
)
from src.league.matchmaker import (
    MatchmakingResult,
    PFSPConfig,
    PFSPMatchmaker,
)
from src.league.registry import (
    LeagueRegistry,
    get_checkpoint_name,
    should_add_to_league,
)
from src.league.types import (
    DEFAULT_BASELINES,
    OPPONENT_TO_AGENT_NAME,
    AgentInfo,
    AgentType,
    LeagueMetadata,
    MatchResult,
    OpponentType,
    opponent_type_to_agent_name,
)

__all__ = [
    # Types
    "OpponentType",
    "AgentType",
    "AgentInfo",
    "MatchResult",
    "MatchmakingResult",
    "LeagueMetadata",
    "DEFAULT_BASELINES",
    "OPPONENT_TO_AGENT_NAME",
    "opponent_type_to_agent_name",
    # Registry
    "LeagueRegistry",
    "should_add_to_league",
    "get_checkpoint_name",
    # Matchmaker
    "PFSPMatchmaker",
    "PFSPConfig",
    # Elo
    "update_elo_multiplayer",
    "update_elo_from_match",
    "batch_update_elo",
    "calculate_expected_score",
    "estimate_win_probability",
    "elo_diff_for_win_rate",
    # Logging
    "log_elo_update",
    "log_league_summary",
    "log_elo_rankings",
    "create_elo_table",
    "create_matchup_heatmap",
]
