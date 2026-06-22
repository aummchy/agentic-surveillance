from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Base class for all agents in the surveillance system.

    All agents follow the same pattern:
    - Accept structured input (dict)
    - Process it
    - Return structured output (dict)

    This makes agents easy to test, swap, and orchestrate.
    """

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's logic.

        Args:
            input_data: Structured input with all data the agent needs.

        Returns:
            Structured output with the agent's decision/result.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
