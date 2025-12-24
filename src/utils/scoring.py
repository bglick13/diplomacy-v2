import numpy as np

from src.engine.wrapper import DiplomacyWrapper


def calculate_leader_gap_penalty(
    game: DiplomacyWrapper,
    power_name: str,
    threshold: int = 3,
) -> float:
    """
    Penalty based on how far behind the leader this power is.

    This encourages powers to stop runaway leaders rather than fighting
    weak neighbors. The penalty only applies when the gap exceeds the threshold.

    Args:
        game: The Diplomacy game wrapper
        power_name: Name of the power to calculate penalty for
        threshold: SC gap above which penalty starts (default 3)

    Returns:
        0 if power is the leader or within threshold.
        Negative value proportional to gap beyond threshold (normalized by 18).
    """
    sc_counts = {pn: len(p.centers) for pn, p in game.game.powers.items()}
    my_scs = sc_counts.get(power_name, 0)
    leader_scs = max(sc_counts.values()) if sc_counts else 0

    gap = leader_scs - my_scs
    if gap <= threshold:
        return 0.0

    # Penalty grows with gap beyond threshold
    # Normalized by 18 (max SCs) so penalty is in [0, 1] range
    excess_gap = gap - threshold
    return -excess_gap / 18.0


def calculate_balance_of_power_score(
    game: DiplomacyWrapper,
    power_name: str,
) -> float:
    """
    Score based on game balance. Higher = more balanced = good for non-leaders.

    Uses inverse coefficient of variation (lower spread = higher score).
    Only positive for non-leaders (leaders should want to break balance, not preserve it).

    This encourages coalition behavior - non-leaders benefit when the game is tight,
    which means they're incentivized to stop anyone from running away.

    Args:
        game: The Diplomacy game wrapper
        power_name: Name of the power to calculate score for

    Returns:
        0 for leaders (they want to break balance).
        0-1 for non-leaders based on game balance (1 = perfectly equal).
    """
    sc_counts = {pn: len(p.centers) for pn, p in game.game.powers.items()}
    my_scs = sc_counts.get(power_name, 0)

    # Filter to surviving powers only
    active_counts = [c for c in sc_counts.values() if c > 0]
    if len(active_counts) < 2:
        return 0.0

    leader_scs = max(active_counts)

    # Leaders don't get balance bonus (they want to break balance)
    if my_scs == leader_scs:
        return 0.0

    # Calculate coefficient of variation (std/mean)
    mean_sc = float(np.mean(active_counts))
    std_sc = float(np.std(active_counts))

    if mean_sc < 0.1:  # Avoid division by zero
        return 0.0

    cv = std_sc / mean_sc  # Higher = more unequal

    # Invert: balance_score = 1 - cv (capped at 0)
    # When game is perfectly equal, cv=0, score=1
    # When one power dominates, cv is high, score approaches 0
    balance_score = max(0.0, 1.0 - cv)

    return balance_score


def calculate_strategic_step_score(
    game: DiplomacyWrapper,
    prev_influence: dict[str, set[str]] | None = None,
    dislodgment_weight: float = 0.5,
    territory_weight: float = 0.2,
    threat_weight: float = 0.3,
    forward_weight: float = 0.1,
) -> dict[str, float]:
    """
    Strategic step scoring that rewards POSITION not SC accumulation.

    This scoring philosophy decouples tactical signals from greedy SC hoarding.
    SC count is only rewarded in final_scores, so the model learns to optimize
    for winning positions rather than immediate territory grab.

    Formula:
    - 0.5 points per Unit (projected power)
    - +0.5 base survival bonus
    - +2.0 if you are the leader (sole or tied)
    - -0.3 per SC gap to leader (if not leader)
    - +(1-cv)*0.5 balance bonus for non-leaders (encourages stopping runaway leaders)
    - +tactical signals (dislodgment, territory, threat, forward)

    Key difference from calculate_step_score: NO SC count reward!
    SC count is only rewarded in final_scores (outcome matters).

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

    # Calculate SC counts for position awareness
    sc_counts = {pn: len(p.centers) for pn, p in game.game.powers.items()}
    leader_scs = max(sc_counts.values()) if sc_counts else 0

    # Calculate balance (coefficient of variation) for non-leader bonus
    active_counts = [c for c in sc_counts.values() if c > 0]
    if len(active_counts) >= 2:
        mean_sc = float(np.mean(active_counts))
        std_sc = float(np.std(active_counts))
        cv = std_sc / mean_sc if mean_sc > 0.1 else 0.0
    else:
        cv = 0.0

    # Build a map of which units are where (for threat detection)
    unit_locations: dict[str, str] = {}  # location -> power_name
    for power_name, power in game.game.powers.items():
        for unit_str in power.units:
            parts = unit_str.split()
            if len(parts) >= 2:
                loc = parts[1].split("/")[0]  # Remove coast
                unit_locations[loc] = power_name

    for power_name, power in game.game.powers.items():
        n_sc = sc_counts.get(power_name, 0)
        n_units = len(power.units)

        # Base score: units + survival (NO SC count!)
        score = (n_units * 0.5) + 0.5  # survival bonus

        # Position awareness: leader bonus / gap penalty
        gap = leader_scs - n_sc
        if gap == 0:
            # We are the leader (or tied for lead)
            score += 2.0
        else:
            # Penalty grows with gap to leader
            score -= gap * 0.3

        # Balance bonus for non-leaders
        # Encourages stopping runaway leaders, coalition behavior
        if n_sc < leader_scs and n_sc > 0:
            # More balanced = better for non-leaders
            # cv=0 (perfectly equal) gives max bonus of 0.5
            # cv=1 (highly unequal) gives 0
            balance_score = max(0.0, 1.0 - cv) * 0.5
            score += balance_score

        # Tactical signals (same as calculate_step_score)

        # 1. Dislodgment signal - per-power attribution
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

        score += n_enemy_dislodged * dislodgment_weight
        score -= n_own_dislodged * dislodgment_weight

        # 2. Territory expansion (influence delta)
        if prev_influence is not None and power_name in prev_influence:
            current_influence = set(power.influence) if hasattr(power, "influence") else set()
            prev_inf = prev_influence.get(power_name, set())
            new_territory = len(current_influence - prev_inf)
            score += new_territory * territory_weight

        # 3. SC threat detection
        n_threatened_scs = 0
        for sc in power.centers:
            try:
                adjacent = game.game.map.abut_list(sc, incl_no_coast=True)
                for adj_loc in adjacent:
                    adj_loc_clean = adj_loc.split("/")[0]
                    if adj_loc_clean in unit_locations:
                        if unit_locations[adj_loc_clean] != power_name:
                            n_threatened_scs += 1
                            break
            except (AttributeError, KeyError):
                pass

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
    use_position_scoring: bool = True,
    solo_victory_sc: int = 18,
    position_bonuses: tuple[float, ...] | None = None,
    elimination_penalty: float = -30.0,
) -> dict[str, float]:
    """
    Computes a heuristic score for every power at the end of a rollout.

    Formula (base):
    - 2.0 points per Supply Center (SC)
    - 0.2 points per Unit (Army/Fleet)
    - 0.1 points per "forward" unit (outside home centers)
    - +0.5 base score for surviving

    Position-based scoring (when use_position_scoring=True):
    - Position bonuses based on SC rank (1st through 7th)
    - Solo victory bonus (full win_bonus) for >= solo_victory_sc
    - Elimination penalty for 0 SCs, 0 units

    Legacy scoring (when use_position_scoring=False):
    - -1.0 points if eliminated
    - +win_bonus if sole leader with >= winner_threshold_sc supply centers

    This position-based approach is inspired by webDiplomacy scoring systems
    (Draw-Size Scoring, Sum-of-Squares). It creates a continuous reward signal
    where every position matters, not just winning.

    Args:
        game: The Diplomacy game wrapper
        win_bonus: Bonus points for solo victory (default 0.0 for backwards compat)
        winner_threshold_sc: Minimum SCs for win bonus in legacy mode
        use_position_scoring: Use position-based bonuses (default True)
        solo_victory_sc: SCs required for solo victory bonus (default 18)
        position_bonuses: Tuple of bonuses for ranks 1-7 (default top-heavy)
        elimination_penalty: Penalty for elimination (default -30.0)

    Returns:
        Dictionary mapping power name to score

    Normalization:
    Scores are raw. The GRPOTrainer handles normalization (subtracting the group mean).
    """
    # Default position bonuses: top-heavy but continuous
    if position_bonuses is None:
        position_bonuses = (50.0, 25.0, 15.0, 10.0, 5.0, 2.0, 0.0)

    map_home_centers = {p: set(game.game.map.homes[p]) for p in game.game.powers}
    scores = {}

    # Calculate SC counts for ranking
    sc_counts = {pn: len(p.centers) for pn, p in game.game.powers.items()}

    # Compute ranks (1 = best, 7 = worst) - handle ties by giving same rank
    # Sort by SC count descending
    sorted_powers = sorted(sc_counts.items(), key=lambda x: x[1], reverse=True)

    # Assign ranks with tie handling
    power_ranks: dict[str, int] = {}
    prev_sc = None
    prev_rank = 0
    for i, (power_name, sc_count) in enumerate(sorted_powers):
        if sc_count == prev_sc:
            # Tie - same rank as previous
            power_ranks[power_name] = prev_rank
        else:
            # New rank
            power_ranks[power_name] = i + 1
            prev_rank = i + 1
        prev_sc = sc_count

    for power_name, power in game.game.powers.items():
        n_sc = len(power.centers)
        n_units = len(power.units)

        # Check elimination
        if n_sc == 0 and n_units == 0:
            if use_position_scoring:
                scores[power_name] = elimination_penalty
            else:
                scores[power_name] = -1.0
            continue

        # Base score: SCs + units + survival bonus
        raw_score = (n_sc * 2.0) + (n_units * 0.2) + 0.5

        # Forward unit bonus
        forward_units = 0
        my_homes = map_home_centers.get(power_name, set())
        for unit_str in power.units:
            parts = unit_str.split()
            if len(parts) >= 2:
                loc = parts[1].split("/")[0]
                if loc not in my_homes:
                    forward_units += 1
        raw_score += forward_units * 0.1

        if use_position_scoring:
            # Position-based scoring
            rank = power_ranks[power_name]

            # Add position bonus (1-indexed, so rank 1 gets index 0)
            if 1 <= rank <= len(position_bonuses):
                raw_score += position_bonuses[rank - 1]

            # Solo victory bonus (on top of 1st place bonus)
            if n_sc >= solo_victory_sc:
                raw_score += win_bonus

        scores[power_name] = raw_score

    # Legacy mode: apply win bonus to sole leader
    if not use_position_scoring and win_bonus > 0:
        max_sc = max(sc_counts.values()) if sc_counts else 0
        if max_sc >= winner_threshold_sc:
            leaders = [pn for pn, sc in sc_counts.items() if sc == max_sc]
            if len(leaders) == 1:
                scores[leaders[0]] += win_bonus

    return scores
