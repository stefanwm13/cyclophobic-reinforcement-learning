#!/usr/bin/env python3

import random
from collections import deque

import cv2
import gym
import numpy as np
from gym import spaces


cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_atari(env_id):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def make_gym_env(env_id):
    return gym.make(env_id)


def wrap_deepmind(
    env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False
):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_pytorch(env):
    return ImageToPyTorch(env)



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
        if self._terminated:
            self.env.reset()
        observation, reward, terminal, info, pos = self.env.step(action)
        #observation, reward, terminal, info = self.env.step(action)

        self._terminated = terminal
        return observation, reward, terminal, info, pos
    
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
        observation, reward, terminal, info, pos = self.env.step(action)
        #observation, reward, terminal, info = self.env.step(action)

        self.total_reward += reward
        self.total_length += 1.0
        if terminal:
            episode_info = dict()
            episode_info['total_reward'] = self.total_reward
            episode_info['total_length'] = self.total_length
            info['episode'] = episode_info
            self.total_reward = 0.0
            self.total_length = 0.0
        return observation, reward, terminal, info, pos

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
        frame = np.asarray(frame, dtype=np.float32)
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

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work.
        Expects inputs to be of shape height x width x num_channels
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        Expects inputs to be of shape num_channels x height x width.
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
