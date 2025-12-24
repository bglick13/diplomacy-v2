"""
TrueSkill rating system for 7-player Diplomacy.

TrueSkill uses (mu, sigma) pairs instead of single Elo numbers:
- mu: Estimated skill level (higher = better)
- sigma: Uncertainty in the estimate (lower = more confident)

Display rating = mu - 3*sigma (conservative 99.7% confidence lower bound)

TrueSkill handles multiplayer games natively without pairwise decomposition,
making it better suited for 7-player Diplomacy than Elo.
"""

from typing import Any

import trueskill

# Default TrueSkill parameters for Diplomacy
# These are the standard TrueSkill defaults
MU_INIT = 25.0  # Initial skill estimate
SIGMA_INIT = 25.0 / 3  # ~8.333, initial uncertainty
BETA = SIGMA_INIT / 2  # ~4.166, performance variance
TAU = SIGMA_INIT / 100  # ~0.0833, skill drift per game
DRAW_PROBABILITY = 0.1  # Probability of ties in score comparisons

# Create a TrueSkill environment with our parameters
env = trueskill.TrueSkill(
    mu=MU_INIT,
    sigma=SIGMA_INIT,
    beta=BETA,
    tau=TAU,
    draw_probability=DRAW_PROBABILITY,
)


def create_rating(mu: float = MU_INIT, sigma: float = SIGMA_INIT) -> trueskill.Rating:
    """Create a TrueSkill rating with the given parameters."""
    return env.create_rating(mu=mu, sigma=sigma)


def compute_display_rating(mu: float, sigma: float) -> float:
    """
    Compute conservative display rating.

    Uses mu - 3*sigma (99.7% confidence lower bound).
    This is the standard TrueSkill "conservative" estimate.

    Args:
        mu: Skill mean
        sigma: Skill uncertainty

    Returns:
        Conservative rating estimate
    """
    return mu - 3 * sigma


def update_trueskill_multiplayer(
    game_results: dict[str, float],
    ratings: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """
    Update TrueSkill ratings based on a multiplayer game result.

    Uses score-based ranking to determine placement. Higher scores = better placement.
    TrueSkill handles multiplayer natively via partial ordering.

    Args:
        game_results: Mapping of agent name to final score
                      e.g., {"adapter_v50": 12.5, "random_bot": 3.0, ...}
        ratings: Current (mu, sigma) ratings for each agent
                 e.g., {"adapter_v50": (28.5, 4.2), "random_bot": (22.1, 6.5), ...}

    Returns:
        Updated (mu, sigma) ratings for each agent
    """
    if len(game_results) < 2:
        return dict(ratings)

    # Sort agents by score (descending) to create ranking
    # TrueSkill expects teams in order of placement (1st, 2nd, 3rd, ...)
    sorted_agents = sorted(game_results.items(), key=lambda x: x[1], reverse=True)

    # Create rating groups - each agent is their own "team"
    # TrueSkill format: [(team1_ratings,), (team2_ratings,), ...]
    rating_groups = []
    for agent, _score in sorted_agents:
        mu, sigma = ratings.get(agent, (MU_INIT, SIGMA_INIT))
        rating_groups.append((env.create_rating(mu=mu, sigma=sigma),))

    # Determine ranks (handle ties by giving same rank)
    ranks = []
    prev_score = None
    prev_rank = 0
    for i, (_agent, score) in enumerate(sorted_agents):
        if prev_score is not None and score == prev_score:
            ranks.append(prev_rank)  # Same rank for tie
        else:
            ranks.append(i)
            prev_rank = i
        prev_score = score

    # Run TrueSkill update
    new_rating_groups = env.rate(rating_groups, ranks=ranks)

    # Extract updated ratings
    new_ratings: dict[str, tuple[float, float]] = {}
    for i, (agent, _score) in enumerate(sorted_agents):
        new_rating = new_rating_groups[i][0]
        new_ratings[agent] = (new_rating.mu, new_rating.sigma)

    return new_ratings


def update_trueskill_from_match(
    power_agents: dict[str, str],
    power_scores: dict[str, float],
    agent_ratings: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """
    Update TrueSkill ratings from a Diplomacy match result.

    This is a convenience wrapper that maps power names to agent names.

    Args:
        power_agents: Mapping of power name to agent name
                      e.g., {"FRANCE": "adapter_v50", "ENGLAND": "random_bot", ...}
        power_scores: Mapping of power name to final score
                      e.g., {"FRANCE": 12.5, "ENGLAND": 3.0, ...}
        agent_ratings: Current (mu, sigma) ratings for each agent

    Returns:
        Updated (mu, sigma) ratings for each agent
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

    return update_trueskill_multiplayer(game_results, agent_ratings)


def batch_update_trueskill(
    matches: list[dict[str, Any]],
    initial_ratings: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """
    Update TrueSkill ratings from a batch of matches.

    Useful for computing rating updates from multiple games at once.

    Args:
        matches: List of match dictionaries with:
                 - power_agents: dict[str, str]
                 - power_scores: dict[str, float]
        initial_ratings: Starting (mu, sigma) ratings

    Returns:
        Final (mu, sigma) ratings after all matches
    """
    current_ratings = dict(initial_ratings)

    for match in matches:
        current_ratings = update_trueskill_from_match(
            power_agents=match["power_agents"],
            power_scores=match["power_scores"],
            agent_ratings=current_ratings,
        )

    return current_ratings


def estimate_win_probability(
    mu_a: float,
    sigma_a: float,
    mu_b: float,
    sigma_b: float,
) -> float:
    """
    Estimate probability that player A beats player B.

    Uses the TrueSkill formula for pairwise win probability.

    Args:
        mu_a: Skill mean of player A
        sigma_a: Skill uncertainty of player A
        mu_b: Skill mean of player B
        sigma_b: Skill uncertainty of player B

    Returns:
        Probability that A beats B (0.0 to 1.0)
    """
    import math

    # Combined uncertainty
    c = math.sqrt(2 * BETA**2 + sigma_a**2 + sigma_b**2)

    # Probability A > B
    return env.cdf((mu_a - mu_b) / c)


def elo_to_trueskill(elo: float) -> tuple[float, float]:
    """
    Convert Elo rating to approximate TrueSkill (mu, sigma).

    Used for backward compatibility with old registries.
    Maps 1000 Elo to 25 mu, with full uncertainty.

    Args:
        elo: Elo rating

    Returns:
        (mu, sigma) tuple
    """
    # Linear mapping: 1000 Elo = 25 mu, 40 Elo points = 1 mu
    mu = (elo - 1000) / 40 + MU_INIT
    # Start with full uncertainty since we don't know how confident the Elo was
    sigma = SIGMA_INIT
    return (mu, sigma)


def trueskill_to_elo(mu: float, sigma: float) -> float:
    """
    Convert TrueSkill rating to approximate Elo.

    Uses display_rating (mu - 3*sigma) for conservative estimate.

    Args:
        mu: Skill mean
        sigma: Skill uncertainty

    Returns:
        Approximate Elo rating
    """
    display = compute_display_rating(mu, sigma)
    # Inverse of elo_to_trueskill mapping
    return (display - MU_INIT) * 40 + 1000
