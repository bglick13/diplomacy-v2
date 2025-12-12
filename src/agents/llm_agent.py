"""
LLM-based Diplomacy Agent

This agent uses an LLM (via vLLM inference) to generate orders.
It encapsulates the prompt engineering and response parsing logic,
making it easy to iterate on prompting strategies.
"""

import json
from dataclasses import dataclass
from typing import Any

from src.engine.wrapper import DiplomacyWrapper


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    # Whether to include chain-of-thought reasoning section
    include_reasoning: bool = True

    # Whether to show all valid moves in the prompt
    show_valid_moves: bool = True
    show_board_context: bool = True

    # Maximum moves to show per unit (to avoid token overflow)
    max_moves_per_unit: int = 10

    # Custom system instructions (appended to base prompt)
    custom_instructions: str = ""

    # Temperature hint for the model (informational, actual temp set in SamplingParams)
    temperature_hint: float = 0.7

    compact_mode: bool = False

    # Optimize prompt structure for vLLM prefix caching
    # When True, static instructions come FIRST so they're shared across all requests
    prefix_cache_optimized: bool = True


# =============================================================================
# Static Prompt Prefixes (for prefix caching)
# =============================================================================
# These are placed at the START of prompts so vLLM can cache the KV values
# and reuse them across all requests. Keep these IDENTICAL across all powers.
#
# Design principles:
# 1. Clear task description upfront
# 2. Rules and format hints (but NO closing tags in example to avoid confusion)
# 3. Dynamic content (power, phase, moves) comes AFTER the static prefix
# 4. Final prompt ends with '<orders>\n' to prime generation
#
# IMPORTANT: Do NOT include '</orders>' in the prefix - model gets confused!

MOVEMENT_PREFIX = """\
### DIPLOMACY MOVEMENT PHASE ###

You are playing Diplomacy. Your task: output exactly one order per unit.

RULES:
- Copy-paste EXACT move strings from the valid moves list below
- One order per line inside <orders> tags
- Movement: "A PAR - BUR" means Army Paris moves to Burgundy
- Hold: "A PAR H" means Army Paris holds
- Support: "A PAR S A BUR - MAR" means Paris supports Burgundy to Marseilles
- Convoy: "F NTH C A LON - NWY" means Fleet convoys Army from London to Norway

"""

# Minimal prefix for when valid moves are NOT shown in prompt
# Model relies on logits processor for valid moves, so we give strategic context only
# CRITICAL: Strong "orders only" instruction to prevent model from generating analysis
# NOTE: Don't use "Move:", "Hold:" prefixes - model copies them literally!
MOVEMENT_PREFIX_MINIMAL = """\
### DIPLOMACY ENGINE ###
ROLE: You are playing the board game Diplomacy.
TASK: Output valid move orders inside <orders> tags.
FORMAT RULES:
1. Output MUST start with <orders>.
2. Output MUST end with </orders>.
3. Do NOT output any text, analysis, or new scenarios after </orders>.
4. One order per line.
Example output:
<orders>
A PAR - BUR
A MAR H
F BRE - MAO
</orders>
"""

ADJUSTMENT_PREFIX = """\
### DIPLOMACY ADJUSTMENT PHASE ###

You are playing Diplomacy in the adjustment phase. Output build or disband orders.

RULES:
- Copy-paste EXACT order strings from the valid options below
- One order per line inside <orders> tags
- Build Army: "A PAR B"
- Build Fleet: "F BRE B"
- Disband: "A PAR D"
- Waive a build: "WAIVE"

"""

ADJUSTMENT_PREFIX_MINIMAL = """\
### DIPLOMACY ORDERS ###

Output orders inside <orders> tags, one per line.
No analysis. No prefixes. Just the orders.

Example output:
<orders>
A PAR B
F BRE B
</orders>

"""


@dataclass
class AgentResponse:
    """Structured response from the LLM agent."""

    raw_text: str
    orders: list[str]
    reasoning: str | None = None

    # Metadata for debugging/observability
    power_name: str = ""
    phase: str = ""
    valid_moves_count: int = 0
    prompt_tokens_estimate: int = 0


class LLMAgent:
    """
    LLM-based agent for playing Diplomacy.

    This class handles:
    1. Prompt construction with configurable strategies
    2. Response parsing and validation
    3. Fallback behavior when parsing fails

    Usage:
        agent = LLMAgent()
        prompt, valid_moves = agent.build_prompt(game, "FRANCE")
        # ... call inference engine with prompt ...
        response = agent.parse_response(raw_output, valid_moves)
    """

    def __init__(self, config: PromptConfig | None = None):
        self.config = config or PromptConfig()

    def build_prompt(
        self, game: DiplomacyWrapper, power_name: str
    ) -> tuple[str, dict[str, list[str]]]:
        """
        Build a prompt for the given power based on phase type.

        Returns:
            (prompt_string, valid_moves_dict)
        """
        valid_moves = game.get_valid_moves(power_name)
        phase = game.get_current_phase()
        phase_type = game.get_phase_type()

        # Get board context for strategic awareness (used in minimal prompt mode)
        board_context = (
            game.get_board_context(power_name) if not self.config.show_valid_moves else None
        )

        # Handle adjustment phases differently
        if phase_type == "A":
            adjustment_delta = game.get_adjustment_delta(power_name)
            prompt = self._construct_adjustment_prompt(
                power_name=power_name,
                phase=phase,
                valid_moves=valid_moves,
                adjustment_delta=adjustment_delta,
                board_context=board_context if self.config.show_board_context else None,
            )
        else:
            # Movement or Retreat phase
            moves_display = self._format_moves_for_prompt(valid_moves)
            example_move = self._get_example_move(valid_moves)
            prompt = self._construct_prompt(
                power_name=power_name,
                phase=phase,
                moves_display=moves_display,
                example_move=example_move,
                valid_moves=valid_moves,
                board_context=board_context if self.config.show_board_context else None,
            )

        return prompt, valid_moves

    def _format_moves_for_prompt(self, valid_moves: dict[str, list[str]]) -> str:
        """Format valid moves as a readable JSON block."""
        # Optionally truncate if too many moves per unit
        truncated = {}
        for unit, moves in valid_moves.items():
            if len(moves) > self.config.max_moves_per_unit:
                truncated[unit] = moves[: self.config.max_moves_per_unit] + [
                    f"... and {len(moves) - self.config.max_moves_per_unit} more"
                ]
            else:
                truncated[unit] = moves
        if self.config.compact_mode:
            return json.dumps(truncated, separators=(",", ":"))
        return json.dumps(truncated, indent=2)

    def _get_example_move(self, valid_moves: dict[str, list[str]]) -> str:
        """Get an example move from the valid moves for the prompt."""
        for _unit, moves in valid_moves.items():
            if moves:
                return moves[0]
        return "A PAR - BUR"  # Fallback

    def _format_board_context(self, board_context: dict) -> str:
        """Format board context concisely (used in minimal mode).

        Keep this data-like, not prose-like, to avoid model generating analysis.
        """
        lines = []

        # Supply centers owned
        my_centers = board_context.get("my_centers", [])
        if my_centers:
            lines.append(f"SCs: {', '.join(sorted(my_centers))}")

        # Neutral centers (if any)
        unowned = board_context.get("unowned_centers", [])
        if unowned:
            lines.append(f"Neutral: {', '.join(sorted(unowned))}")

        # Opponent positions (very compact)
        opponent_units = board_context.get("opponent_units", {})
        if opponent_units:
            opp_parts = []
            for power, units in sorted(opponent_units.items()):
                opp_parts.append(f"{power}:{','.join(units)}")
            lines.append(f"Opponents: {' | '.join(opp_parts)}")

        return "\n".join(lines)

    def _construct_prompt(
        self,
        power_name: str,
        phase: str,
        moves_display: str,
        example_move: str,
        valid_moves: dict[str, list[str]],
        board_context: dict | None = None,
    ) -> str:
        """Construct the full prompt string.

        IMPORTANT: The prompt ends with '<orders>\n' to prime the model.
        This ensures the logits processor immediately enters ACTIVE mode
        and constrains generation to valid moves from the trie.

        When prefix_cache_optimized=True, static instructions come FIRST
        so vLLM can cache and reuse them across all requests.

        When show_valid_moves=False, we only show unit positions (not exhaustive
        move lists). The logits processor handles validity, so we save tokens.
        Board context provides strategic info but is clearly separated from orders.
        """

        # Count total units for context
        unit_count = len(valid_moves)
        unit_list = ", ".join(valid_moves.keys())  # e.g., "A PAR, F BRE, A MAR"

        if self.config.compact_mode:
            if self.config.prefix_cache_optimized:
                if self.config.show_valid_moves:
                    # Full mode: show all valid moves
                    prompt = (
                        f"{MOVEMENT_PREFIX}"
                        f"GAME STATE:\n"
                        f"Power: {power_name}\n"
                        f"Phase: {phase}\n"
                        f"You have {unit_count} units.\n\n"
                        f"VALID MOVES:\n{moves_display}\n\n"
                        f"Output {unit_count} orders (one per unit):\n"
                        "<orders>\n"
                    )
                else:
                    # Minimal mode: unit positions + board context for strategy
                    # Logits processor handles move validity
                    # Key: Strong "orders only" framing to prevent analysis generation
                    board_info = self._format_board_context(board_context) if board_context else ""
                    prompt = (
                        f"{MOVEMENT_PREFIX_MINIMAL}"
                        f"SITUATION:\n"
                        f"You are {power_name}. Phase: {phase}\n"
                        f"Your units: {unit_list}\n"
                        f"{board_info}\n"
                        f"OUTPUT {unit_count} ORDERS:\n"
                        "<orders>\n"
                    )
            else:
                # Legacy compact format (dynamic content first)
                prompt = (
                    f"You are {power_name} in a game of Diplomacy. Phase:{phase}. Units:{unit_count}. "
                    "ValidMoves JSON follows"
                    f"{moves_display}"
                    "Emit exactly one order per unit using EXACT strings. Output format: <orders>...\n</orders>"
                    "<orders>\n"
                )
        else:
            if self.config.show_valid_moves:
                prompt = f"""You are playing Diplomacy as {power_name}.
    Phase: {phase}
    Units: {unit_count}

    Your units and their valid moves:
    {moves_display}

    Output exactly {unit_count} orders, one per line.
    Use the EXACT move strings from the valid moves list above.

    Example:
    <orders>
    A PAR - BUR
    F BRE - MAO
    A MAR H
    </orders>
    """
            else:
                board_info = self._format_board_context(board_context) if board_context else ""
                prompt = f"""You are {power_name}. Phase: {phase}
Your units: {unit_list}
{board_info}

Output ONLY {unit_count} orders (no analysis):
<orders>
"""

        if self.config.custom_instructions:
            prompt += f"\n{self.config.custom_instructions}\n"

        if not self.config.compact_mode:
            prompt += "<orders>\n"

        return prompt

    def _construct_adjustment_prompt(
        self,
        power_name: str,
        phase: str,
        valid_moves: dict[str, list[str]],
        adjustment_delta: int,
        board_context: dict | None = None,
    ) -> str:
        """Construct prompt for adjustment phases (builds/disbands).

        During adjustment phases:
        - adjustment_delta > 0: Power can BUILD units
        - adjustment_delta < 0: Power must DISBAND units
        - adjustment_delta == 0: No action needed (but may still have WAIVE option)
        """
        moves_display = self._format_moves_for_prompt(valid_moves)

        if adjustment_delta > 0:
            # BUILD phase
            action = "BUILD"
            order_count = adjustment_delta
            example = self._get_example_move(valid_moves)
            if not example or example == "A PAR - BUR":
                example = "A PAR B"  # Default build example
        elif adjustment_delta < 0:
            # DISBAND phase
            action = "DISBAND"
            order_count = abs(adjustment_delta)
            example = self._get_example_move(valid_moves)
            if not example or example == "A PAR - BUR":
                example = "A PAR D"  # Default disband example
        else:
            # No action needed, but we still generate a prompt
            # (power may choose to WAIVE or the engine handles it)
            if not valid_moves:
                # No orders to give
                return f"""You are playing Diplomacy as {power_name}.
Phase: {phase}

No orders required this phase. Your supply centers equal your units.
<orders>
WAIVE
"""
            action = "ADJUST"
            order_count = len(valid_moves)
            example = self._get_example_move(valid_moves)

        if self.config.compact_mode:
            if self.config.prefix_cache_optimized:
                if self.config.show_valid_moves:
                    # Full mode: show all valid options
                    prompt = (
                        f"{ADJUSTMENT_PREFIX}"
                        f"GAME STATE:\n"
                        f"Power: {power_name}\n"
                        f"Phase: {phase}\n"
                        f"Action: {action} {order_count} unit(s)\n\n"
                        f"VALID OPTIONS:\n{moves_display}\n\n"
                        f"Output {order_count} order(s):\n"
                        "<orders>\n"
                    )
                else:
                    # Minimal mode: action info + board context
                    # For adjustment, valid moves list is usually small so savings are modest
                    board_info = self._format_board_context(board_context) if board_context else ""
                    prompt = (
                        f"{ADJUSTMENT_PREFIX_MINIMAL}"
                        f"SITUATION:\n"
                        f"You are {power_name}. Phase: {phase}\n"
                        f"Action: {action} {order_count} unit(s)\n"
                        f"{board_info}\n"
                        f"OUTPUT {order_count} ORDER(S):\n"
                        "<orders>\n"
                    )
            else:
                # Legacy compact format (dynamic content first)
                prompt = (
                    f"You are {power_name} in {phase}. Action:{action} count:{order_count}. "
                    "Valid options JSON follows. Emit EXACT adjustment orders in <orders> "
                    "immediately with no explanation.\n"
                    f"{moves_display}\n<orders>\n"
                )
        else:
            if self.config.show_valid_moves:
                prompt = f"""You are playing Diplomacy as {power_name}.
Phase: {phase}
Action: {action} {order_count} unit(s)

Available locations and orders:
{moves_display}

Output exactly {order_count} order(s), one per line.
Use the EXACT order strings from the list above.

Order format:
- Build Army: A <LOC> B
- Build Fleet: F <LOC> B
- Disband: A <LOC> D or F <LOC> D
- Skip build: WAIVE

Example:
<orders>
{example}
</orders>
"""
            else:
                # Minimal mode for non-compact
                prompt = f"""You are playing Diplomacy as {power_name}.
Phase: {phase}
Action: {action} {order_count} unit(s)

Output exactly {order_count} order(s), one per line.
- Build Army: A <HOME_SC> B (e.g., A PAR B)
- Build Fleet: F <COASTAL_HOME_SC> B (e.g., F BRE B)
- Disband: <UNIT> D (e.g., A MUN D)
- Skip build: WAIVE

<orders>
"""

        if self.config.custom_instructions:
            prompt += f"\n{self.config.custom_instructions}\n"

        # Prime the model
        if not self.config.compact_mode:
            prompt += "<orders>\n"

        return prompt

    def parse_response(
        self,
        raw_text: str,
        valid_moves: dict[str, list[str]],
        power_name: str = "",
        phase: str = "",
    ) -> AgentResponse:
        """
        Parse the LLM response and extract orders.

        Args:
            raw_text: Raw output from the LLM
            valid_moves: Dict of valid moves for validation
            power_name: Power name for metadata
            phase: Game phase for metadata

        Returns:
            AgentResponse with parsed orders and metadata
        """
        from src.utils.parsing import extract_orders

        orders = extract_orders(raw_text)

        # Count valid moves for metadata
        total_valid = sum(len(moves) for moves in valid_moves.values())

        return AgentResponse(
            raw_text=raw_text,
            orders=orders,
            power_name=power_name,
            phase=phase,
            valid_moves_count=total_valid,
            prompt_tokens_estimate=len(raw_text.split()),  # Rough estimate
        )

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        """
        Synchronous interface matching DiplomacyAgent protocol.

        NOTE: This requires external inference. For actual use, call
        build_prompt() and parse_response() separately with your inference engine.
        """
        raise NotImplementedError(
            "LLMAgent requires async inference. Use build_prompt() and parse_response() instead."
        )

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        """Return empty press (no diplomacy messages)."""
        return []


class LLMAgentWithFallback(LLMAgent):
    """
    LLM Agent with fallback to a baseline agent when parsing fails.

    This is useful during development to ensure games can complete
    even when the LLM produces unparseable output.
    """

    def __init__(
        self,
        config: PromptConfig | None = None,
        fallback_agent: Any | None = None,
    ):
        super().__init__(config)

        if fallback_agent is None:
            from src.agents.baselines import RandomBot

            fallback_agent = RandomBot()

        self.fallback_agent = fallback_agent

    def parse_response_with_fallback(
        self,
        raw_text: str,
        valid_moves: dict[str, list[str]],
        game: DiplomacyWrapper,
        power_name: str,
    ) -> AgentResponse:
        """
        Parse response, falling back to baseline agent if no orders extracted.
        """
        response = self.parse_response(raw_text, valid_moves, power_name, game.get_current_phase())

        # If no orders extracted, use fallback
        if not response.orders:
            fallback_orders = self.fallback_agent.get_orders(game, power_name)
            response.orders = fallback_orders
            response.reasoning = "[FALLBACK] LLM produced no parseable orders"

        return response


# =============================================================================
# Prompt Templates for Different Strategies
# =============================================================================


def get_aggressive_config() -> PromptConfig:
    """Config that encourages aggressive play."""
    return PromptConfig(
        custom_instructions="""
STRATEGY HINT: Prioritize movement orders (using " - ") over holds and supports.
Expansion is key in the early game. Be aggressive!
""",
    )


def get_defensive_config() -> PromptConfig:
    """Config that encourages defensive play."""
    return PromptConfig(
        custom_instructions="""
STRATEGY HINT: Prioritize support orders to defend your positions.
Hold your ground and wait for opportunities.
""",
    )


def get_balanced_config() -> PromptConfig:
    """Default balanced config."""
    return PromptConfig()
