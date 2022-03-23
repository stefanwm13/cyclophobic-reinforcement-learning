"""Contains `wrappers` that can wrap around environments to modify their functionality.

The implementations of these wrappers are adopted from
`OpenAI <https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py>`_.

.. autosummary::
   :toctree: wrappers
"""

import time

import cv2
import gym
import numpy as np

cv2.ocl.setUseOpenCL(False)  # do not use OpenCL


class AutoResetWrapper(gym.Wrapper):
    """A wrapper that automatically resets the environment in case of termination.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self._terminated = False

    def step(self, action):
        reset_obs = None
        if self._terminated:
            reset_obs = self.env.reset()
        #observation, reward, terminal, info, pos = self.env.step(action)
        observation, reward, terminal, info = self.env.step(action)

        self._terminated = terminal
        return observation, reward, terminal, info
    
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._terminated = False
        return observation


class ClipRewardWrapper(gym.RewardWrapper):
    """A wrapper that clips the rewards between -1 and 1.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.clip(reward, -1., 1.)


class EpisodeInfoWrapper(gym.Wrapper):
    """A wrapper that stores episode information in the `info` returned by  :meth:`gym.Env.step`
    at the end of an episode. More specifically, if an episode is terminal,
    `info` will contain the key `'episode'` which has a :obj:`dict` value containing the `'total_reward'` ,
    which is the cumulative reward of the episode.

    Note
    ----
        If you want to get the cumulative reward of the entire episode,
        an :py:class:`EpisodicLifeWrapper` (like :py:class:`~testing_suite.environment.atari_wrappers:AtariEpisodicLifeWrapper`)
        should be used *after* this wrapper.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self.total_reward = 0.0
        self.total_length = 0.0

    def step(self, action):
        #observation, reward, terminal, info, pos = self.env.step(action)
        observation, reward, terminal, info = self.env.step(action)

        self.total_reward += reward
        self.total_length += 1.0
        if terminal:
            episode_info = dict()
            episode_info['total_reward'] = self.total_reward
            episode_info['total_length'] = self.total_length
            info['episode'] = episode_info
            self.total_reward = 0.0
            self.total_length = 0.0
        return observation, reward, terminal, info#, pos

    def reset(self, **kwargs):
        self.total_reward = 0.0
        self.total_length = 0.0
        return self.env.reset(**kwargs)

    @staticmethod
    def get_episode_rewards_from_info_batch(infos):
        """Utility function that extracts the episode rewards,
        that are inserted by :py:class:`EpisodeInfoWrapper`, out of the *infos*.

        Calls :py:meth:`~EpisodeInfoWrapper.get_episode_key_from_info_batch`

        Arguments
        ---------
            infos: list
                A batch-major list of `infos` as generated during interaction
                with an environment wrapped with :py:class:`EpisodeInfoWrapper`

        Returns
        -------
            rewards: :py:obj:`np.ndarray`
                A batch-major array with the same shape as infos.
                It contains the episode reward of an `info` at the corresponding position.
                If no episode reward was in an `info` , the result will contain :py:obj:`numpy.nan` respectively.
        """
        return EpisodeInfoWrapper.get_episode_key_from_info_batch(infos, 'total_reward')

    @staticmethod
    def get_episode_lengths_from_info_batch(infos):
        """Utility function that extracts the episode lengths,
        that are inserted by, :py:class:`EpisodeInfoWrapper` out of the *infos*.

        Calls :py:meth:`~EpisodeInfoWrapper.get_episode_key_from_info_batch`

        Arguments
        ---------
            infos: list
                A batch-major list of `infos` as generated during interaction
                with an environment wrapped with :py:class:`EpisodeInfoWrapper`

        Returns
        -------
            lengths: :py:obj:`np.ndarray`
                A batch-major array with the same shape as infos.
                It contains the episode length of an `info` at the corresponding position.
                If no episode length was in an `info` , the result will contain :py:obj:`numpy.nan` respectively.
        """
        return EpisodeInfoWrapper.get_episode_key_from_info_batch(infos, 'total_length')

    @staticmethod
    def get_solved_from_info_batch(infos):
        """Utility function that extracts info, which episodes were correctly solved.
        Only works with gym_sokoban environments. That information is inserted by :py:class:`EpisodeInfoWrapper`.

        Arguments
        ---------
            infos : list
                A batch-major list of `infos` as generated during interaction
                with an environment wrapped with :py:class:`EpisodeInfoWrapper`

        Returns
        -------
            solves: :py:obj:`np.ndarray`
                A batch-major array with the same shape as infos.
                It contains the `episode solved` values of an `info` at the corresponding position.
        """
        solved = np.full_like(infos, np.nan, np.float32)
        if solved.ndim == 1:
            steps = len(solved)
            for step in range(steps):
                info = infos[step]

                if 'maxsteps_used' in info.keys():
                    if info['all_boxes_on_target']:
                        solved[step] = 1.
                    elif info['maxsteps_used']:
                        solved[step] = 0.

        elif solved.ndim == 2:
            environments, steps = solved.shape
            for environment in range(environments):
                for step in range(steps):
                    info = infos[environment][step]

                    if 'maxsteps_used' in info.keys():
                        if info['all_boxes_on_target']:
                            solved[environment, step] = 1.
                        elif info['maxsteps_used']:
                            solved[environment, step] = 0.
        return solved

    @staticmethod
    def get_episode_key_from_info_batch(infos, key):
        """Utility function that extracts a given `key` , out of the `infos`.
        That information is inserted by :py:class:`EpisodeInfoWrapper`.

        Arguments
        ---------
            infos: list
                A batch-major list of `infos` as generated during interaction
                with an environment wrapped with :py:class:`EpisodeInfoWrapper`
            key: str
                The key marking the wanted info. Currently works for keys:
                    * 'total_length'
                    * 'total_reward'

        Returns
        -------
            data_array: :py:obj:`np.ndarray`
                It contains the episodes information linked to given `key` at the corresponding position of an `info`.
        """

        data_array = np.full_like(infos, np.nan, np.float32)
        if data_array.ndim == 1:
            steps = len(data_array)
            for step in range(steps):
                info = infos[step]

                if 'episode' in info:
                    data_key = info['episode'][key]
                    data_array[step] = data_key
        elif data_array.ndim == 2:
            environments, steps = data_array.shape
            for environment in range(environments):
                for step in range(steps):
                    info = infos[environment][step]

                    if 'episode' in info:
                        data_key = info['episode'][key]
                        data_array[environment, step] = data_key

        return data_array


class FrameStackWrapper(gym.Wrapper):
    """A wrapper that stacks the last observations.
    The observations returned by this wrapper consist of the last frames.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
        num_stacked_frames: int
            The number of frames that will be stacked.
    """

    def __init__(self, env, num_stacked_frames):
        super().__init__(env)
        self._num_stacked_frames = num_stacked_frames

        stacked_low = np.repeat(env.observation_space.low, num_stacked_frames, axis=-1)
        stacked_high = np.repeat(env.observation_space.low, num_stacked_frames, axis=-1)
        self.observation_space = gym.spaces.Box(low=stacked_low, high=stacked_high, dtype=env.observation_space.dtype)

        self._stacked_frames = np.zeros_like(stacked_low)

    def step(self, action):
        next_frame, reward, terminal, info = self.env.step(action)
        self._stacked_frames = np.roll(self._stacked_frames, shift=-1, axis=-1)
        if terminal:
            self._stacked_frames.fill(0.0)
        self._stacked_frames[..., -1:] = next_frame
        return self._stacked_frames, reward, terminal, info

    def reset(self, **kwargs):
        frame = self.env.reset(**kwargs)
        self._stacked_frames = np.repeat(frame, self._num_stacked_frames, axis=-1)
        return self._stacked_frames


class PreprocessFrameWrapper(gym.ObservationWrapper):
    """A wrapper that scales the observations from 210x160 to :py:attr:`width` x :py:attr:`height`.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
        width: int
            The target width of a given observation
        height: int
            The target height of a given observation
    """

    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
        self.dim = self.observation_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, self.dim), dtype=np.float32)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)  # scale
        #frame = np.asarray(frame, dtype=np.float32)
        return frame


class RenderWrapper(gym.Wrapper):
    """A wrapper that calls :py:meth:`gym.Env.render` every step.

    Arguments
    ---------
        env: :py:obj:`gym.Env`
            An environment that will be wrapped.
        fps: int, float, optional
            If it is not None, the steps will be slowed down
            to run at the specified frames per second by waiting 1.0/ `fps` seconds every step.
    """

    def __init__(self, env, fps=None):
        super().__init__(env)
        self._spf = 1.0 / fps if fps is not None else None

    def step(self, action):
        self.env.render()
        if self._spf is not None:
            time.sleep(self._spf)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
