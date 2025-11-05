import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import load_env_creator
import random
import gc

import torch
import torch.nn.functional as F

from environments.mujoco import rand_param_envs

"""
Custom wrappers for VariBAD meta-RL framework.

Note on Gymnasium compatibility:
- VariBadWrapper: Custom multi-episode BAMDP wrapper (no gymnasium equivalent)
- TimeLimitMask: Thin wrapper to add 'bad_transition' info (gymnasium.wrappers.TimeLimit handles truncation)
- MetaWorldSparseRewardWrapper: Custom sparse reward conversion (domain-specific)

Gymnasium already provides:
- NormalizeObservation, NormalizeReward: For observation/reward normalization
- TimeLimit: For episode time limits (we add TimeLimitMask on top for 'bad_transition' marking)
- RecordVideo: For video recording
- Various other wrappers in gymnasium.wrappers
"""

try:
    # this is to suppress some warnings (in the newer mujoco versions)
    gym.logger.set_level(40)
except AttributeError:
    pass


def mujoco_wrapper(entry_point, **kwargs):
    """
    Load the environment from its entry point.
    In gymnasium, use load_env_creator instead of load.
    """
    env_creator = load_env_creator(entry_point)
    env = env_creator(**kwargs)
    return env


class VariBadWrapper(gym.Wrapper):
    def __init__(self, env, episodes_per_task, env_type, args):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - adding the done info to the state (might be needed to make states markov)
        """

        super().__init__(env)

        self.env_type = env_type
        self.args = args

        # make sure we can call these attributes even if the orig env does not have them
        if not hasattr(self.env.unwrapped, "task_dim"):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, "belief_dim"):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, "get_belief"):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, "num_states"):
            self.env.unwrapped.num_states = None
        if not hasattr(
            self.env.unwrapped, "_max_episode_steps"
        ):  # Meta-World ML10/ML45
            self.env.unwrapped._max_episode_steps = env.max_path_length

        if episodes_per_task > 1:
            self.add_done_info = True
        else:
            self.add_done_info = False

        if self.add_done_info:
            if isinstance(self.observation_space, spaces.Box) or isinstance(
                self.observation_space, rand_param_envs.gym.spaces.box.Box
            ):
                if len(self.observation_space.shape) > 1:
                    raise ValueError  # can't add additional info for obs of more than 1D
                self.observation_space = spaces.Box(
                    low=np.array([*self.observation_space.low, 0]),
                    # shape will be deduced from this
                    high=np.array([*self.observation_space.high, 1]),
                )
            else:
                # Not implemented. Would need to add something simliar for the other possible spaces,
                # "Space", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"
                raise NotImplementedError

        # calculate horizon length H^+
        self.episodes_per_task = episodes_per_task
        # counts the number of episodes
        self.episode_count = 0

        # count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # the horizon in the BAMDP is the one in the MDP times the number of episodes per task,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = self.episodes_per_task * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = (
                self.episodes_per_task * self.env.unwrapped._max_episode_steps
            )

        # add dummy timesteps in-between episodes for resetting the MDP
        self.horizon_bamdp += self.episodes_per_task - 1

        # this tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def reset(self, task=None, seed=None, options=None):
        """Resets the BAMDP"""

        try:  # meta-world cannot take task spec
            self.env.reset_task(task)
        except (TypeError, AttributeError):
            try:
                self.env.reset_task()
            except AttributeError:
                pass  # env doesn't have reset_task

        # Gymnasium returns (obs, info)
        try:
            state, info = self.env.reset(seed=seed, options=options)
        except AttributeError:
            state, info = self.env.unwrapped.reset(seed=seed, options=options)

        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))

        return state, info

    def reset_mdp(self, seed=None):
        """Resets the underlying MDP only (*not* the task)."""
        state, info = self.env.reset(seed=seed)
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state

    def step(self, action):
        # Gymnasium returns (obs, reward, terminated, truncated, info)
        state, reward, terminated, truncated, info = self.env.step(action)
        self.done_mdp = terminated or truncated

        if self.env_type == "metaworld":
            if self.env._max_episode_steps == self.env.curr_path_length:
                self.done_mdp = True
                info["bad_transition"] = True

        info["done_mdp"] = self.done_mdp

        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))

        self.step_count_bamdp += 1
        # if we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes in this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_task:
                done_bamdp = True

        if self.done_mdp and not done_bamdp:
            if self.env_type == "Maze":
                # In minecraft and tmaze envs, it is necessary to see start state of next ep, but not terminal state
                info["term_state"] = state
                state = self.reset_mdp()
                if self.add_done_info:
                    state[-1] = 1.0
                info["start_state"] = state
            else:
                info["start_state"] = self.reset_mdp()

        # Return 5-tuple for gymnasium compatibility
        # In BAMDP, terminated means all episodes in task are done
        return state, reward, done_bamdp, False, info

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        (This one's only needed for mujoco 131)
        """
        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            try:
                orig_attr = self.env.__getattribute__(attr)
            except AttributeError:
                orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


class TimeLimitMask(gym.Wrapper):
    """
    Wrapper to mark episodes that end due to time limits.
    In gymnasium, TimeLimit wrapper sets truncated=True, but we need to add
    'bad_transition' to info for compatibility with RL algorithms.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Mark time limit truncations as 'bad_transition' for value bootstrapping
        if truncated:
            info["bad_transition"] = True

        # Return 5-tuple for gymnasium compatibility
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info  # Return tuple for gymnasium compatibility


class MetaWorldSparseRewardWrapper(gym.Wrapper):
    """
    Wrapper to convert MetaWorld dense rewards to sparse (success-based) rewards.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(info.get("success", 0.0))
        return obs, reward, terminated, truncated, info
