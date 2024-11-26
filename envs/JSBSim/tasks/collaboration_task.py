"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-19 14:27:03
LastEditTime : 2024-11-26 21:53:23
FilePath     : /LAG/envs/JSBSim/tasks/collaboration_task.py
Description  : 
"""

import torch
import numpy as np
from gymnasium import spaces
from collections import deque

from .single_combat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import (
    AltitudeReward,
    CollaborationReward,
    EventDrivenReward,
    RelativeAltitudeReward,
    ShootPenaltyReward,
)
from ..core.simulator import MissileSimulator
from ..utils.utils import LLA2NEU, get_AO_TA_R, get_root_dir
from ..model.baseline_actor import BaselineActor


class CollaborationTask(SingleCombatTask):
    def __init__(self, config):
        super().__init__(config)
        self.max_attack_angle = getattr(self.config, "max_attack_angle", 180)
        self.max_attack_distance = getattr(self.config, "max_attack_distance", np.inf)
        self.min_attack_interval = getattr(self.config, "min_attack_interval", 125)
        # NOTE(zzp): Finish Reward
        self.reward_functions = [
            AltitudeReward(self.config),
            CollaborationReward(self.config),
            EventDrivenReward(self.config),
            RelativeAltitudeReward(self.config),
            ShootPenaltyReward(self.config),
        ]

    def reset(self, env):
        """Reset fighter blood & missile status"""
        # TODO(zzp): shoot action for enm?
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self._last_shoot_time = {
            agent_id: -self.min_attack_interval for agent_id in env.agents.keys()
        }
        self.remaining_missiles = {
            agent_id: agent.num_missiles for agent_id, agent in env.agents.items()
        }
        self.lock_duration = {
            agent_id: deque(maxlen=int(1 / env.time_interval))
            for agent_id in env.agents.keys()
        }
        return super().reset(env)

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10.0, shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        # TODO(zzp): continuous or discrete action space?
        self.action_space = spaces.Tuple(
            [spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)]
        )

    def normalize_action(self, env, agent_id, action):
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space
        TODO(zzp): use absolute/relative enm info and missile info
        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - First missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(
            env.agents[agent_id].get_property_values(self.state_var)
        )
        enm_obs_list = np.array(
            env.agents[agent_id].enemies[0].get_property_values(self.state_var)
        )
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(
            *ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt
        )
        enm_cur_ned = LLA2NEU(
            *enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt
        )
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(
            ego_feature, enm_feature, return_side=True
        )
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate(
                (missile_sim.get_position(), missile_sim.get_velocity())
            )
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(
                ego_feature, missile_feature, return_side=True
            )
            norm_obs[15] = (
                np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]
            ) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag

    def step(self, env):
        super().step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = (
                agent.is_alive
                and self._shoot_action[agent_id]
                and self.remaining_missiles[agent_id] > 0
            )
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(
                        parent=agent, target=agent.enemies[0], uid=new_missile_uid
                    )
                )
                self.remaining_missiles[agent_id] -= 1


class HierarchicalCollaborationTask(HierarchicalSingleCombatTask, CollaborationTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(config)
        # TODO(zzp): model device in HierarchicalSingleCombatTask
        # TODO(zzp): How about angles?
        self.reward_functions = [
            AltitudeReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config),
        ]

    def load_observation_space(self):
        return CollaborationTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        # TODO(zzp): Difference between no-Hierarchical
        self.action_space = spaces.Tuple(
            [spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)]
        )

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action."""
        self._shoot_action[agent_id] = action[-1]
        return HierarchicalSingleCombatTask.normalize_action(
            self, env, agent_id, action[:-1].astype(np.int32)
        )

    def get_obs(self, env, agent_id):
        return CollaborationTask.get_obs(self, env, agent_id)

    def reset(self, env):
        """Task-specific reset, include reward function reset."""
        self._inner_rnn_states = {
            agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()
        }
        CollaborationTask.reset(self, env)

    def step(self, env):
        CollaborationTask.step(self, env)
