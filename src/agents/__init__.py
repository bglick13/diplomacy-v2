"""
Diplomacy Agents

This module provides various agent implementations for playing Diplomacy:
- RandomBot: Baseline that picks random valid moves
- ChaosBot: Aggressive baseline that prioritizes movement
- LLMAgent: LLM-based agent with configurable prompting
"""

from src.agents.base import DiplomacyAgent
from src.agents.baselines import ChaosBot, RandomBot
from src.agents.llm_agent import (
    AgentResponse,
    LLMAgent,
    LLMAgentWithFallback,
    PromptConfig,
    get_aggressive_config,
    get_balanced_config,
    get_defensive_config,
)

__all__ = [
    "DiplomacyAgent",
    "RandomBot",
    "ChaosBot",
    "LLMAgent",
    "LLMAgentWithFallback",
    "PromptConfig",
    "AgentResponse",
    "get_aggressive_config",
    "get_balanced_config",
    "get_defensive_config",
]
