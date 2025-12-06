from src.engine.wrapper import DiplomacyWrapper


def calculate_final_scores(
    game: DiplomacyWrapper,
    win_bonus: float = 0.0,
    winner_threshold_sc: int = 10,
) -> dict[str, float]:
    """
    Computes a heuristic score for every power at the end of a rollout.

    Formula:
    - 2.0 points per Supply Center (SC)
    - 0.2 points per Unit (Army/Fleet)
    - 0.1 points per "forward" unit (outside home centers)
    - +0.5 base score for surviving
    - -1.0 points if eliminated (0 units, 0 SCs)
    - +win_bonus if sole leader with >= winner_threshold_sc supply centers

    The win_bonus creates pressure to WIN outright, not just survive. This is
    critical for breaking cooperative stalemate equilibria in self-play.

    Args:
        game: The Diplomacy game wrapper
        win_bonus: Bonus points for the winning power (default 0.0 for backwards compat)
        winner_threshold_sc: Minimum supply centers to be eligible for win bonus

    Returns:
        Dictionary mapping power name to score

    Normalization:
    Scores are raw. The GRPOTrainer handles normalization (subtracting the group mean).
    """
    map_home_centers = {p: set(game.game.map.homes[p]) for p in game.game.powers}
    scores = {}

    for power_name, power in game.game.powers.items():
        # 1. Check Elimination
        n_sc = len(power.centers)
        n_units = len(power.units)

        if n_sc == 0 and n_units == 0:
            scores[power_name] = -1.0
            continue

        # 2. Compute Weighted Score
        # SCs are the primary objective (Real power)
        # Units are secondary (Projected power)
        raw_score = (n_sc * 2.0) + (n_units * 0.2) + 0.5

        forward_units = 0
        my_homes = map_home_centers.get(power_name, set())

        for unit_str in power.units:
            # Format is usually "A PAR" or "F SPA/SC"
            parts = unit_str.split()
            if len(parts) < 2:
                continue

            # Extract location (remove coast if present)
            loc = parts[1].split("/")[0]

            # If the unit is occupying territory that is NOT a home center, reward it
            if loc not in my_homes:
                forward_units += 1

        # Weight this highly to force the model to explore openings
        raw_score += forward_units * 0.1

        scores[power_name] = raw_score

    # Apply win bonus to the sole leader (if any)
    if win_bonus > 0:
        # Find the power(s) with the most supply centers
        sc_counts = {pn: len(p.centers) for pn, p in game.game.powers.items()}
        max_sc = max(sc_counts.values()) if sc_counts else 0

        if max_sc >= winner_threshold_sc:
            # Find all powers with max SCs (could be tied)
            leaders = [pn for pn, sc in sc_counts.items() if sc == max_sc]

            # Only award bonus if there's a SOLE leader (no ties)
            if len(leaders) == 1:
                winner = leaders[0]
                scores[winner] += win_bonus

    return scores
