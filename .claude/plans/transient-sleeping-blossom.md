# DumbBot Port: Stronger Baseline for Modal Training

## Context

For long training runs, `base_model` Elo will plateau. We need stronger baselines to:
1. Continue measuring learning progress
2. Match research benchmarks (DumbBot is standard)

**Constraint**: Must run on Modal (in-memory games, no DAIDE servers).

---

## DumbBot Algorithm (from C++ source)

Source: [diplomacy/daide-client](https://github.com/diplomacy/daide-client/tree/master/bots/dumbbot)

### Stage 1: Province Value Calculation

For each province/coast, calculate a value based on:

```python
# Base value depends on ownership
if is_our_supply_center:
    base_value = size_of_largest_adjacent_enemy_power  # Defense value
elif is_enemy_supply_center:
    base_value = size_of_owning_power  # Attack value
else:
    base_value = 0  # Non-SC

# Proximity values (recursive blur)
proximity[0] = base_value * weight_proximity_0
for n in range(1, N_PROXIMITY_LEVELS):
    proximity[n] = sum(adjacent_proximity[n-1]) / 5

# Competition and strength
strength = count_adjacent_friendly_units
competition = max_adjacent_enemy_units_from_single_power

# Final coast value
value = (
    sum(proximity * weight_proximity) +
    strength * weight_strength +
    competition * weight_competition
)
```

### Stage 2: Move Generation

```python
for unit in randomized(my_units):
    # Calculate destination values
    destinations = {}
    for move in valid_moves[unit]:
        dest = extract_destination(move)
        destinations[dest] = coast_values[dest]

    # Probabilistic selection (not purely greedy)
    if best_dest == current_location:
        order = HOLD
    elif dest_is_occupied_by_friendly:
        if friendly_needs_support:
            order = SUPPORT(friendly_unit)
        else:
            order = MOVE(second_best_dest)
    else:
        order = MOVE(best_dest)
```

### Weight Parameters (from source)

Original weights were chosen by intuition, not optimized:
- Proximity weights: distance-decaying
- Strength weight: small positive
- Competition weight: small negative

---

## Implementation Plan

### Python Port Structure

```python
class DumbBot(DiplomacyAgent):
    """
    Port of David Norman's DumbBot algorithm.

    Two-stage process:
    1. Calculate province values (defense, attack, proximity blur)
    2. Generate orders based on values (with coordination logic)
    """

    # Tunable weights (original values from C++ source)
    PROXIMITY_WEIGHTS = [100, 50, 25, 12, 6]  # Distance decay
    STRENGTH_WEIGHT = 10
    COMPETITION_WEIGHT = -20

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        # Stage 1: Calculate province values
        province_values = self._calculate_province_values(game, power_name)

        # Stage 2: Generate orders
        return self._generate_orders(game, power_name, province_values)
```

### Key Methods to Implement

| Method | Purpose | ~Lines |
|--------|---------|--------|
| `_calculate_province_values()` | Base values + proximity blur | 50 |
| `_get_power_size()` | Count units for a power | 10 |
| `_get_adjacent_provinces()` | Map adjacency lookup | 15 |
| `_calculate_strength()` | Friendly adjacent units | 10 |
| `_calculate_competition()` | Max enemy adjacent units | 15 |
| `_generate_orders()` | Move selection with coordination | 60 |
| `_select_destination()` | Probabilistic value-weighted selection | 20 |
| `_handle_retreat()` | Retreat phase logic | 20 |
| `_handle_build()` | Build/disband phase logic | 20 |

**Total**: ~220 lines

### Helper Data Needed

The `diplomacy` package provides:
- `game.game.powers[power].units` - Unit locations
- `game.game.powers[power].centers` - Supply centers owned
- `game.get_valid_moves(power)` - Valid moves per unit
- `game.game.map.loc_abut` - Adjacency information

### Coordination Logic

```python
def _generate_orders(self, game, power_name, values):
    orders = []
    claimed_destinations = set()

    for unit in randomized(my_units):
        valid = game.get_valid_moves(power_name)[unit]

        # Score each possible destination
        dest_scores = {}
        for move in valid:
            dest = self._extract_destination(move)
            if dest not in claimed_destinations:
                dest_scores[move] = values.get(dest, 0)

        # Select best unclaimed destination
        best_move = max(dest_scores, key=dest_scores.get)
        best_dest = self._extract_destination(best_move)

        # Check for support opportunities
        if self._should_support(best_dest, claimed_destinations):
            support_move = self._find_support_move(unit, valid, best_dest)
            if support_move:
                orders.append(support_move)
                continue

        claimed_destinations.add(best_dest)
        orders.append(best_move)

    return orders
```

---

## Files to Modify

```
src/agents/baselines.py        # Add DumbBot class (~220 lines)
src/apps/rollouts/app.py       # Add "dumb_bot" to BASELINE_BOTS dict (1 line)
tests/test_dumbbot.py          # Unit tests for value calculation (new file)
```

---

## Success Criteria

1. DumbBot compiles and runs without errors
2. DumbBot generates valid orders (no DATC violations)
3. DumbBot Elo stabilizes higher than CoordinatedBot
4. DumbBot is beatable by trained checkpoints (not too strong)

---

## References

- [DumbBot C++ source](https://github.com/diplomacy/daide-client/tree/master/bots/dumbbot)
- [DAIDE DumbBot Algorithm](http://www.daide.org.uk/s0003.html) (archived)
- [SearchBot Paper](https://ar5iv.labs.arxiv.org/html/2010.02923) - uses Albert as benchmark
