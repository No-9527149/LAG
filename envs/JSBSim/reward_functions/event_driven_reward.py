"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-26 19:50:46
FilePath     : /LAG/envs/JSBSim/reward_functions/event_driven_reward.py
Description  : 
"""

from .reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """Return rewards when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200

    Args:
        BaseRewardFunction (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env, agent_id, info={}):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        if env.agents[agent_id].is_shot_down:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 200
        for missile in env.agents[agent_id].launch_missiles:
            # TODO(zzp): break or ?
            if missile.is_success:
                reward += 200
        return self._process(reward, agent_id)
