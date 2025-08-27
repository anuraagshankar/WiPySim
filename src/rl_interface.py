from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Decision:
    ap_id: int
    kind: str  # "joint" for SARL
    obs: np.ndarray
    valid_actions: list[int]


class ExternalRLInterface:
    """
    Simple broker to bridge SimPy-driven MACs and an external RL driver (e.g., RLlib).

    - MAC publishes a Decision (non-blocking) via publish_decision.
    - The environment wrapper polls pending decisions and supplies actions via provide_action.
    - For each AP, an action callback is registered that the interface calls to apply the action
      inside the simulator (e.g., MAC.apply_external_action).
    """

    def __init__(self):
        self._pending: dict[int, Decision] = {}
        self._action_handlers: dict[int, Callable[[int], None]] = {}

    def register_agent(self, ap_id: int, action_handler: Callable[[int], None]):
        self._action_handlers[ap_id] = action_handler

    def publish_decision(
        self, ap_id: int, kind: str, obs: np.ndarray, valid_actions: list[int]
    ):
        self._pending[ap_id] = Decision(ap_id=ap_id, kind=kind, obs=obs, valid_actions=valid_actions)

    def has_pending(self) -> bool:
        return len(self._pending) > 0

    def consume_pending(self) -> dict[int, Decision]:
        # Caller may want a snapshot without clearing; we return and keep state.
        return dict(self._pending)

    def clear_pending(self, ap_id: int):
        self._pending.pop(ap_id, None)

    def provide_action(self, ap_id: int, action: int):
        if ap_id not in self._action_handlers:
            raise KeyError(f"No action handler registered for AP {ap_id}")
        # Clear pending before applying to avoid double-consume race in wrappers
        self.clear_pending(ap_id)
        self._action_handlers[ap_id](action)


