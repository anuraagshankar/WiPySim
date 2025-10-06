"""
RLlib PPO trainer for the MultiAPEnv multi-agent environment.

Usage examples:
  - Train with defaults (CPU only):
      python -m src.components.rllib_agents --iters 50

  - Customize workers and batch sizes:
      python -m src.components.rllib_agents --iters 100 --num-workers 2 \
          --train-batch-size 8000 --sgd-minibatch-size 256 --num-sgd-iter 20

All AP agents share a single PPO policy (homogeneous agents) and are
mapped via a simple shared policy mapping function.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

from src.envs.rllib_multi_ap import MultiAPEnv


ENV_NAME = "MultiAPEnv-v0"
SHARED_POLICY_ID = "shared_policy"

# -----------------------
# Training configuration
# -----------------------
DEFAULT_ITERS: int = 50
NUM_WORKERS: int = 0
NUM_GPUS: int = 0
LR: float = 5e-4
GAMMA: float = 0.99
TRAIN_BATCH_SIZE: int = 4000
SGD_MINIBATCH_SIZE: int = 128
NUM_SGD_ITER: int = 10
FCNET_HIDDENS: Tuple[int, int] = (128, 128)
CHECKPOINT_EVERY: int = 10  # set to 0 to disable periodic checkpoints


def env_creator(_: Dict[str, Any] | None = None) -> MultiAPEnv:
    # MultiAPEnv handles its own internal config via UserConfig/SimParams
    # provided in src.envs.rllib_multi_ap.
    return MultiAPEnv()


def build_ppo_config(
    num_workers: int,
    num_gpus: int,
    lr: float,
    gamma: float,
    train_batch_size: int,
    sgd_minibatch_size: int,
    num_sgd_iter: int,
    fcnet_hiddens: Tuple[int, int],
) -> PPOConfig:
    # Instantiate a temporary env instance to infer spaces for the shared policy
    temp_env = env_creator({})
    obs_map, _ = temp_env.reset()
    # Pick one agent to obtain spaces; all APs are homogeneous
    example_agent = next(iter(obs_map.keys()))
    obs_space = temp_env.get_observation_space(example_agent)
    act_space = temp_env.get_action_space(example_agent)

    def policy_mapping_fn(_agent_id: Any, _episode: Any, _worker: Any, **_kwargs) -> str:
        return SHARED_POLICY_ID

    config = (
        PPOConfig()
        .environment(env=ENV_NAME, disable_env_checking=True)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .rollouts(num_rollout_workers=num_workers)
        .training(
            lr=lr,
            gamma=gamma,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            model={
                "fcnet_hiddens": list(fcnet_hiddens),
                "fcnet_activation": "tanh",
            },
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .multi_agent(
            policies={
                SHARED_POLICY_ID: (
                    None,  # default PPO policy class
                    obs_space,
                    act_space,
                    {},
                )
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[SHARED_POLICY_ID],
        )
    )
    return config


def main():
    # Register env
    register_env(ENV_NAME, env_creator)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Build and train PPO
    config = build_ppo_config(
        num_workers=NUM_WORKERS,
        num_gpus=NUM_GPUS,
        lr=LR,
        gamma=GAMMA,
        train_batch_size=TRAIN_BATCH_SIZE,
        sgd_minibatch_size=SGD_MINIBATCH_SIZE,
        num_sgd_iter=NUM_SGD_ITER,
        fcnet_hiddens=FCNET_HIDDENS,
    )

    algo = config.build()

    last_checkpoint = None
    for i in range(1, DEFAULT_ITERS + 1):
        result = algo.train()
        ep_reward_mean = result.get("episode_reward_mean")
        train_timesteps = result.get("timesteps_total")
        print(
            f"Iter {i:04d} | mean_reward={ep_reward_mean} | timesteps={train_timesteps}"
        )

        if CHECKPOINT_EVERY and i % CHECKPOINT_EVERY == 0:
            last_checkpoint = algo.save().checkpoint_path
            print(f"Saved checkpoint: {last_checkpoint}")

    if last_checkpoint is None:
        last_checkpoint = algo.save().checkpoint_path
        print(f"Saved final checkpoint: {last_checkpoint}")


if __name__ == "__main__":
    main()


