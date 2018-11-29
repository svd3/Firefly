from .firelfy_task import Model
from .env_utils import pos_init
from .env_variables import *

from gym.envs.registration import register

from .gym_input import true_params

register(
    id ='FireflyTorch-v0',
    entry_point ='FireflyEnv.firefly_gym:FireflyEnv',
)
