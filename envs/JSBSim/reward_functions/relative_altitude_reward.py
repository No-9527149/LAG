"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-26 19:51:40
FilePath     : /LAG/envs/JSBSim/reward_functions/relative_altitude_reward.py
Description  : 
"""

import numpy as np
from .reward_function_base import BaseRewardFunction


class RelativeAltitudeReward(BaseRewardFunction):
    """Punish if current fighter does not satisfy constraints. Typically negative.
    - Punishment of relative altitude when larger than 1000 (range: [-1, 0])

    Args:
        BaseRewardFunction (_type_): _description_

    NOTE:
    - Only for one-on-one environment
    """

    def __init__(self, config):
        super().__init__(config)
        self.KH = getattr(self.config, f"{self.__class__.__name__}_KH", 1.0)  # km

    def get_reward(self, task, env, agent_id, info={}):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # unit: km
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000  # unit: km
        new_reward = min(self.KH - np.abs(ego_z - enm_z), 0)
        return self._process(new_reward, agent_id)
