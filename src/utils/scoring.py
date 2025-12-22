from src.engine.wrapper import DiplomacyWrapper


def calculate_step_score(
    game: DiplomacyWrapper,
    prev_influence: dict[str, set[str]] | None = None,
    dislodgment_weight: float = 0.5,
    territory_weight: float = 0.2,
    threat_weight: float = 0.3,
    forward_weight: float = 0.1,
) -> dict[str, float]:
    """
    Compute a richer score for each power based on current board state.

    Used for per-step reward shaping to provide immediate feedback on decisions.
    This includes tactical signals that change during movement phases, not just
    build phases (when SC counts change).

    Formula:
    - 2.0 points per Supply Center (SC)
    - 0.3 points per Unit (Army/Fleet)
    - +dislodgment_weight per enemy unit dislodged
    - -dislodgment_weight per own unit dislodged
    - +territory_weight per new province in influence (vs prev_influence)
    - -threat_weight per SC threatened by adjacent enemy unit
    - +forward_weight per unit outside home centers

    Args:
        game: The Diplomacy game wrapper
        prev_influence: Optional dict of power -> set of provinces in influence
                        before this step. Used to calculate territory delta.
        dislodgment_weight: Weight for dislodgment signals (default 0.5)
        territory_weight: Weight for territory expansion (default 0.2)
        threat_weight: Weight for SC threats (default 0.3)
        forward_weight: Weight for forward unit positioning (default 0.1)

    Returns:
        Dictionary mapping power name to score
    """
    scores = {}
    map_home_centers = {p: set(game.game.map.homes[p]) for p in game.game.powers}

    # Build a map of which units are where (for threat detection)
    unit_locations: dict[str, str] = {}  # location -> power_name
    for power_name, power in game.game.powers.items():
        for unit_str in power.units:
            parts = unit_str.split()
            if len(parts) >= 2:
                loc = parts[1].split("/")[0]  # Remove coast
                unit_locations[loc] = power_name

    for power_name, power in game.game.powers.items():
        n_sc = len(power.centers)
        n_units = len(power.units)

        # Base score: SCs and units
        score = (n_sc * 2.0) + (n_units * 0.3)

        # 1. Dislodgment signal - per-power attribution
        # +reward for dislodging enemy, -reward for being dislodged
        n_own_dislodged = 0
        n_enemy_dislodged = 0

        if hasattr(game.game, "dislodged") and game.game.dislodged:
            for dislodged_unit, attacker_site in game.game.dislodged.items():
                # Find owner of dislodged unit
                dislodged_owner = None
                for pn, p in game.game.powers.items():
                    if dislodged_unit in p.units:
                        dislodged_owner = pn
                        break

                # Find attacker unit at that site
                attacker_unit = (
                    game.game._occupant(attacker_site) if hasattr(game.game, "_occupant") else None
                )
                attacker_owner = None
                if attacker_unit:
                    for pn, p in game.game.powers.items():
                        if attacker_unit in p.units:
                            attacker_owner = pn
                            break

                # Attribute to current power
                if dislodged_owner == power_name:
                    n_own_dislodged += 1  # We got dislodged (bad)
                if attacker_owner == power_name:
                    n_enemy_dislodged += 1  # We dislodged someone (good)

        # Apply rewards: +bonus for dislodging, -penalty for being dislodged
        score += n_enemy_dislodged * dislodgment_weight
        score -= n_own_dislodged * dislodgment_weight

        # 2. Territory expansion (influence delta)
        if prev_influence is not None and power_name in prev_influence:
            current_influence = set(power.influence) if hasattr(power, "influence") else set()
            prev_inf = prev_influence.get(power_name, set())
            new_territory = len(current_influence - prev_inf)
            score += new_territory * territory_weight

        # 3. SC threat detection
        # A SC is "threatened" if an adjacent tile has an enemy unit
        n_threatened_scs = 0
        for sc in power.centers:
            # Get adjacent locations
            try:
                adjacent = game.game.map.abut_list(sc, incl_no_coast=True)
                for adj_loc in adjacent:
                    adj_loc_clean = adj_loc.split("/")[0]
                    if adj_loc_clean in unit_locations:
                        if unit_locations[adj_loc_clean] != power_name:
                            n_threatened_scs += 1
                            break  # Only count each SC once
            except (AttributeError, KeyError):
                pass  # Skip if map data unavailable

        score -= n_threatened_scs * threat_weight

        # 4. Forward unit positioning
        my_homes = map_home_centers.get(power_name, set())
        forward_units = 0
        for unit_str in power.units:
            parts = unit_str.split()
            if len(parts) >= 2:
                loc = parts[1].split("/")[0]
                if loc not in my_homes:
                    forward_units += 1

        score += forward_units * forward_weight

        scores[power_name] = score

    return scores


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
