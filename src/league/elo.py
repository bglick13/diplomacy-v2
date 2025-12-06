"""
Multi-player Elo calculation for 7-player Diplomacy.

Standard Elo is designed for 1v1 games. For Diplomacy, we use a pairwise
decomposition approach: each game is treated as 21 pairwise matchups
(7 choose 2), and Elo updates are scaled accordingly.
"""

from typing import Any


def calculate_expected_score(elo_a: float, elo_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Uses the standard Elo formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        elo_a: Elo rating of player A
        elo_b: Elo rating of player B

    Returns:
        Expected score for player A (0.0 to 1.0)
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def calculate_actual_score(score_a: float, score_b: float) -> float:
    """
    Calculate actual score for pairwise comparison.

    Args:
        score_a: Final game score for player A
        score_b: Final game score for player B

    Returns:
        1.0 if A > B, 0.5 if tied, 0.0 if A < B
    """
    if score_a > score_b:
        return 1.0
    elif score_a == score_b:
        return 0.5
    else:
        return 0.0


def update_elo_multiplayer(
    game_results: dict[str, float],
    elo_ratings: dict[str, float],
    k: float = 32.0,
) -> dict[str, float]:
    """
    Update Elo ratings based on a multiplayer game result.

    Uses pairwise decomposition: for each pair of players, we compute
    the expected and actual scores, then update both players' Elo.
    The K-factor is scaled by 1/(n-1) where n is the number of players
    to account for multiple pairwise comparisons.

    Args:
        game_results: Mapping of agent name to final score
                      e.g., {"adapter_v50": 12.5, "random_bot": 3.0, ...}
        elo_ratings: Current Elo ratings for each agent
                     e.g., {"adapter_v50": 1200, "random_bot": 800, ...}
        k: Base K-factor for Elo updates (default 32)

    Returns:
        Updated Elo ratings for each agent
    """
    agents = list(game_results.keys())
    n_players = len(agents)

    if n_players < 2:
        return dict(elo_ratings)

    # Initialize new Elo ratings
    new_elos: dict[str, float] = dict(elo_ratings.items())

    # Scale K-factor by number of pairwise comparisons per player
    # Each player is involved in (n-1) pairwise comparisons
    k_scaled = k / (n_players - 1)

    # Process all pairwise matchups
    for i, agent_a in enumerate(agents):
        for agent_b in agents[i + 1 :]:
            # Get current Elo ratings
            elo_a = elo_ratings.get(agent_a, 1000.0)
            elo_b = elo_ratings.get(agent_b, 1000.0)

            # Calculate expected and actual scores
            expected_a = calculate_expected_score(elo_a, elo_b)
            actual_a = calculate_actual_score(game_results[agent_a], game_results[agent_b])

            # Update both players
            delta = k_scaled * (actual_a - expected_a)
            new_elos[agent_a] = new_elos.get(agent_a, elo_a) + delta
            new_elos[agent_b] = new_elos.get(agent_b, elo_b) - delta

    return new_elos


def update_elo_from_match(
    power_agents: dict[str, str],
    power_scores: dict[str, float],
    agent_elos: dict[str, float],
    k: float = 32.0,
) -> dict[str, float]:
    """
    Update Elo ratings from a Diplomacy match result.

    This is a convenience wrapper that maps power names to agent names.

    Args:
        power_agents: Mapping of power name to agent name
                      e.g., {"FRANCE": "adapter_v50", "ENGLAND": "random_bot", ...}
        power_scores: Mapping of power name to final score
                      e.g., {"FRANCE": 12.5, "ENGLAND": 3.0, ...}
        agent_elos: Current Elo ratings for each agent
        k: K-factor for Elo updates

    Returns:
        Updated Elo ratings for each agent
    """
    # Build game_results by mapping power scores to agent names
    game_results: dict[str, float] = {}

    for power_name, agent_name in power_agents.items():
        score = power_scores.get(power_name, 0.0)

        # If the same agent plays multiple powers, sum their scores
        if agent_name in game_results:
            game_results[agent_name] += score
        else:
            game_results[agent_name] = score

    # Average scores if an agent plays multiple powers
    agent_power_counts: dict[str, int] = {}
    for agent_name in power_agents.values():
        agent_power_counts[agent_name] = agent_power_counts.get(agent_name, 0) + 1

    for agent_name, count in agent_power_counts.items():
        if count > 1:
            game_results[agent_name] /= count

    return update_elo_multiplayer(game_results, agent_elos, k=k)


def batch_update_elo(
    matches: list[dict[str, Any]],
    initial_elos: dict[str, float],
    k: float = 32.0,
) -> dict[str, float]:
    """
    Update Elo ratings from a batch of matches.

    Useful for computing Elo updates from multiple games at once
    (e.g., in the async Elo evaluator).

    Args:
        matches: List of match dictionaries with:
                 - power_agents: dict[str, str]
                 - power_scores: dict[str, float]
        initial_elos: Starting Elo ratings
        k: K-factor for Elo updates

    Returns:
        Final Elo ratings after all matches
    """
    current_elos = dict(initial_elos)

    for match in matches:
        current_elos = update_elo_from_match(
            power_agents=match["power_agents"],
            power_scores=match["power_scores"],
            agent_elos=current_elos,
            k=k,
        )

    return current_elos


def estimate_win_probability(elo_a: float, elo_b: float) -> float:
    """
    Estimate probability that player A beats player B.

    This is equivalent to expected score in 1v1 context.

    Args:
        elo_a: Elo rating of player A
        elo_b: Elo rating of player B

    Returns:
        Probability that A beats B (0.0 to 1.0)
    """
    return calculate_expected_score(elo_a, elo_b)


def elo_diff_for_win_rate(target_win_rate: float) -> float:
    """
    Calculate Elo difference needed for a target win rate.

    Useful for understanding what Elo gaps mean in practice.

    Args:
        target_win_rate: Desired win rate (0.0 to 1.0)

    Returns:
        Elo difference needed (positive means higher rating)

    Examples:
        - elo_diff_for_win_rate(0.75) ≈ 191  (75% win rate)
        - elo_diff_for_win_rate(0.90) ≈ 382  (90% win rate)
        - elo_diff_for_win_rate(0.99) ≈ 798  (99% win rate)
    """
    if target_win_rate <= 0.0 or target_win_rate >= 1.0:
        raise ValueError("Win rate must be between 0 and 1 (exclusive)")

    # Solve: target_win_rate = 1 / (1 + 10^(-diff/400))
    # => 10^(-diff/400) = 1/target_win_rate - 1
    # => -diff/400 = log10(1/target_win_rate - 1)
    # => diff = -400 * log10(1/target_win_rate - 1)
    import math

    return -400.0 * math.log10(1.0 / target_win_rate - 1.0)
