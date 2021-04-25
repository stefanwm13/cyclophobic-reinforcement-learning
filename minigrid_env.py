"""Contains `wrappers` that can wrap around minigrid_ environments to modify their functionality.

.. _minigrid: https://github.com/maximecb/gym-minigrid

.. autosummary::
   :toctree: minigrid
"""

from gym_minigrid import wrappers as mg_wrappers

from helpers import *

cv2.ocl.setUseOpenCL(False)  # do not use OpenCL


def make_minigrid_env(env_id, render=False, **kwargs):
    """Creates a :py:obj:`gym.Env` and wraps it.

    Used wrappers:
        #. :py:class:`~mg_wrappers.RGBImgObsWrapper`
        #. :py:class:`~mg_wrappers.FullyObsWrapper`
        #. :py:class:`~testing_suite.environment.wrappers.wrappers.PreprocessFrameWrapper`
            using :py:const:`~testing_suite.environment.wrappers.wrappers.PreprocessFrameWrapper.width` = 84
            and :py:const:`~testing_suite.environment.wrappers.wrappers.PreprocessFrameWrapper.height` = 84
        #. :py:class:`~testing_suite.environment.wrappers.wrappers.EpisodeInfoWrapper`
        #. :py:class:`~testing_suite.environment.wrappers.wrappers.AutoResetWrapper`

    Arguments
    ---------
        env_id: str
            The id of the gym environment to be created.
        render: bool
            Boolean flag that activates rendering for recording episodes (not implemented yet)

    Returns
    -------
        env: :py:obj:`gym.Env`
            The generated (and wrapped) environment
    """
    env = gym.make(env_id)                      # create a gym environment

    env = mg_wrappers.RGBImgObsWrapper(env)     # extract the rgb image observation from the default state dict
    env = mg_wrappers.FullyObsWrapper(env)      # use fully observable states
    env = mg_wrappers.ImgObsWrapper(env)        # use image output as observation space
    env = PreprocessFrameWrapper(env, 84, 84)   # preprocess (scale down) the observations
    env = EpisodeInfoWrapper(env)               # stores episode info in 'info' at the end of an episode
    env = AutoResetWrapper(env)                 # reset on terminated episode

    return env
