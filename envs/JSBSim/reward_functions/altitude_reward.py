"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-11 17:53:46
FilePath     : /LAG/envs/JSBSim/reward_functions/altitude_reward.py
Description  : 
"""

import numpy as np
from .reward_function_base import BaseRewardFunction


class AltitudeReward(BaseRewardFunction):
    """Penalties will be imposed if current aircraft does not meet certain constraints.
    Typically negative:
    - Penalty when the altitude is lower than the safe altitude (range: [-1, 0])
    - Penalty when the altitude is lower than the danger altitude (range: [-1, 0])

    Args:
        BaseRewardFunction (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(
            self.config, f"{self.__class__.__name__}_safe_altitude", 4.0
        )  # km
        self.danger_altitude = getattr(
            self.config, f"{self.__class__.__name__}_danger_altitude", 3.5
        )  # km
        self.Kv = getattr(self.config, f"{self.__class__.__name__}_Kv", 0.2)  # mh
        # TODO(zzp): _Pv, _PH?
        self.reward_item_names = [
            self.__class__.__name__ + item for item in ["", "_Pv", "_PH"]
        ]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000  # unit: km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340  # unit: mh
        Pv = 0.0
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(
                ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude,
                0.0,
                1.0,
            )
        PH = 0.0
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0.0, 1.0) - 1.0 - 1.0
        new_reward = Pv + PH
        return self._process(new_reward, agent_id, (Pv, PH))
