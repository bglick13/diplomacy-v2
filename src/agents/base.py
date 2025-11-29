from typing import Any, Dict, List, Protocol

from src.engine.wrapper import DiplomacyWrapper


class DiplomacyAgent(Protocol):
    """
    Protocol that all agents must adhere to.
    """

    def get_orders(self, game: DiplomacyWrapper, power_name: str) -> List[str]:
        """
        Given the game state and the power to play, return a list of valid orders.
        """
        ...

    def get_press(
        self, game: DiplomacyWrapper, power_name: str
    ) -> List[Dict[str, Any]]:
        """
        (Optional) Return messages to send.
        For baseline bots, this usually returns [].
        """
        ...
