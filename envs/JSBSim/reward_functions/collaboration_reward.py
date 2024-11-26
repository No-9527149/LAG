"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-25 19:17:26
LastEditTime : 2024-11-26 21:49:12
FilePath     : /LAG/envs/JSBSim/reward_functions/collaboration_reward.py
Description  : 
"""

import math
import numpy as np
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R, get_pincer_angle


class CollaborationReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        # max_ao: math.pi (rad)
        # max_ta: math.pi (rad)
        # BUG: math.pi
        # TODO(zzp): config
        self.max_ao = getattr(
            self.config, f"{self.__class__.__name__}_max_ao", "math.pi"
        )
        self.max_ta = getattr(
            self.config, f"{self.__class__.__name__}_max_ta", "math.pi"
        )
        self.reward_item_names = [
            self.__class__.__name__ + item
            for item in ["aa_ata", "pincer", "pre_angle", "velocity"]
        ]

    def get_reward(self, task, env, agent_id, info={}):
        reward = 0
        ego = env.agents[agent_id]
        ego_feature = np.hstack([ego.get_position(), ego.get_velocity()])
        enm = env.agents[agent_id].enemies[0]
        enm_feature = np.hstack([enm.get_position(), enm.get_velocity()])
        missile = env.agents[agent_id].launch_missiles[0]
        missile_feature = np.hstack([missile.get_position(), missile.get_velocity()])

        ao, ta, r = get_AO_TA_R(ego_feature, enm_feature)
        ao_ta_reward = self.get_ao_ta_reward(ao, ta)

        pincer_angle = get_pincer_angle(ego_feature, enm_feature, missile_feature)
        pincer_reward = self.get_pincer_reward(pincer_angle)

        current_step = info["current_step"]
        buffer_size = self.config["all_args"].buffer_size
        pre_angle_reward = self.get_pre_angle_reward(
            ao, ta, pincer_angle, current_step, buffer_size
        )

        velocity_reward = self.get_velocity_reward(ego_feature, enm_feature)
        # TODO(zzp): scale
        return ao_ta_reward + pincer_angle + pre_angle_reward + velocity_reward

    def get_ao_ta_reward(self, ao, ta):
        # [0, 2]
        reward_rr, reward_rb = 0, 0
        if ao <= self.max_ao:
            reward_rr = math.exp(-ao / self.max_ao)
        if ta > self.max_ta:
            reward_rb = math.exp(-ta / self.max_ta)
        # TODO(zzp): clip
        return reward_rr + reward_rb

    def get_pincer_reward(self, pincer_angle):
        # [-1, e^{1 / 2} - 1]
        # [-1, 2.3]
        reward_pincer = (
            2 * math.exp(-np.abs(pincer_angle - math.pi / 2) / self.max_ao) - 1
        )
        # TODO(zzp): clip
        return reward_pincer

    def get_pre_angle_reward(self, ao, ta, pincer_angle, current_step, buffer_size):
        # TODO(zzp): clip
        reward_pre_angle = -1
        if current_step <= buffer_size // 3 and ta < math.pi / 6:
            reward_pre_angle = 2 * math.exp(-6 * ta / self.max_ao) - 1
        elif (
            buffer_size // 3 < current_step < 2 * buffer_size // 3
            and pincer_angle < math.pi / 5
        ):
            reward_pre_angle = 2 * math.exp(-5 * pincer_angle / self.max_ao) - 1
        elif ao < math.pi / 5:
            reward_pre_angle = 2 * math.exp(-5 * ao / self.max_ao) - 1
        return reward_pre_angle

    def get_velocity_reward_fn(self, ego_feature, enm_feature):
        # [-1, 1]
        _, _, _, ego_vx, ego_vy, ego_vz = ego_feature
        _, _, _, enm_vx, enm_vy, enm_vz = enm_feature
        ego_v = np.array([ego_vx, ego_vy, ego_vz])
        enm_v = np.array([enm_vx, enm_vy, enm_vz])
        reward_vel = np.clip(ego_v - enm_v, -1.0, 1.0)
        return reward_vel
