from typing import Any, Protocol

from src.engine.wrapper import DiplomacyWrapper


class DiplomacyAgent(Protocol):
    """
    Protocol that all agents must adhere to.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> list[str]:
        """
        Given the game state and the power to play, return a list of valid orders.
        """
        ...

    def get_press(self, game: DiplomacyWrapper, power_name: str) -> list[dict[str, Any]]:
        """
        (Optional) Return messages to send.
        For baseline bots, this usually returns [].
        """
        ...
