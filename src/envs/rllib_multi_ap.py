from __future__ import annotations

from typing import Dict, Any

import numpy as np
import simpy
from gymnasium.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.user_config import UserConfig as cfg_module
from src.sim_params import SimParams as sparams_module
from src.utils.support import initialize_network
from src.rl_interface import ExternalRLInterface
from src.components.rl_agents import CHANNEL_MAP, CW_MAP


class MultiAPEnv(MultiAgentEnv):
    """
    RLlib-style multi-agent environment where each RL-driven AP is an agent.

    Agent IDs: f"ap_{ap.id}"
    Observation: 9-dim joint SARL context per AP (Box[0,1])
    Action: Discrete over SARL joint valid actions index space (shared size across APs)
    Reward: per-AP negative sum of delays from last tx attempt
    """

    def __init__(self, cfg=cfg_module, sparams=sparams_module):
        super().__init__()
        self.cfg = cfg
        self.sparams = sparams
        self._env = None
        self._iface = None
        self._net = None
        self._ap_agents: Dict[str, Any] = {}
        self._ap_id_by_name: Dict[str, int] = {}
        self._prev_acted: set[str] = set()
        # Shared spaces across agents
        self._obs_space = Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)
        # Compute joint action count following SARLController
        self._valid_joint_actions = [
            (c_id, p - 1, cw_id)
            for c_id, pset in CHANNEL_MAP.items()
            for p in pset
            for cw_id in CW_MAP.keys()
        ]
        self._act_space = Discrete(len(self._valid_joint_actions))

    @staticmethod
    def _agent_name(ap_id: int) -> str:
        return f"ap_{ap_id}"

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._env = simpy.Environment()
        if seed is not None:
            # simpy doesn't use numpy RNG directly; keep consistency using Random in initialize_network
            pass
        # touch options to avoid linter warnings if provided
        if options:
            _ = options
        self._iface = ExternalRLInterface()
        self._net = initialize_network(self.cfg, self.sparams, self._env)

        # Register RL-driven APs
        self._ap_agents.clear()
        self._ap_id_by_name.clear()
        for ap in self._net.get_aps():
            if ap.mac_layer.rl_driven:
                ap.mac_layer.enable_external_control(self._iface)
                name = self._agent_name(ap.id)
                self._ap_agents[name] = ap
                self._ap_id_by_name[name] = ap.id

        # Step until first set of decisions is pending or sim ends
        observations = {}
        self._run_until_pending_or_end()
        for ap_name, ap in self._ap_agents.items():
            pending = self._iface.consume_pending()
            if ap.id in pending:
                observations[ap_name] = pending[ap.id].obs.astype(np.float32)
        self._prev_acted = set()
        return observations, {}

    def _advance_one_event(self):
        self._env.step()

    def _is_truncated(self) -> bool:
        return self._env.now >= self.cfg.SIMULATION_TIME_us

    def _run_until_pending_or_end(self):
        while not self._iface.has_pending() and not self._is_truncated():
            self._advance_one_event()

    def step(self, action_dict: Dict[str, int]):
        # Feed actions to the corresponding APs
        for agent_name, action in action_dict.items():
            ap = self._ap_agents.get(agent_name)
            if ap is None:
                continue
            self._iface.provide_action(ap.id, int(action))

        # Run until next pending or end
        self._run_until_pending_or_end()

        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        # Observations for agents requiring a decision now
        pending = self._iface.consume_pending()
        for agent_name, ap in self._ap_agents.items():
            if ap.id in pending:
                observations[agent_name] = pending[ap.id].obs.astype(np.float32)

        # Rewards for APs that acted in the previous call
        for agent_name, ap in self._ap_agents.items():
            rewards[agent_name] = (
                float(getattr(ap.mac_layer, "last_reward", 0.0))
                if agent_name in self._prev_acted
                else 0.0
            )
            terminateds[agent_name] = False
            truncateds[agent_name] = False
            infos[agent_name] = {}

        # Track who acted this step (to reward next time)
        self._prev_acted = set(action_dict.keys())

        env_truncated = self._is_truncated()
        terminateds["__all__"] = False
        truncateds["__all__"] = env_truncated

        return observations, rewards, terminateds, truncateds, infos

    # RLlib API spaces per-agent
    def get_observation_space(self, agent_id: Any):
        return self._obs_space

    def get_action_space(self, agent_id: Any):
        return self._act_space




if __name__ == "__main__":
    # Simple demo: sample random valid actions until the episode ends
    env = MultiAPEnv()
    obs_map, info_map = env.reset()
    step_idx = 0
    print(f"Reset: {list(obs_map.keys())} agents ready")
    while True:
        # Sample a random action for each agent that is currently ready
        actions = {}
        for agent_key in obs_map.keys():
            act_space = env.get_action_space(agent_key)
            actions[agent_key] = int(act_space.sample())

        obs_map, rewards_map, terminateds_map, truncateds_map, infos_map = env.step(actions)
        step_idx += 1

        # Print minimal progress
        if step_idx % 10 == 0:
            print(f"Step {step_idx}: rewards snapshot: {rewards_map}")

        if terminateds_map.get("__all__", False) or truncateds_map.get("__all__", False):
            print("Episode finished.")
            break