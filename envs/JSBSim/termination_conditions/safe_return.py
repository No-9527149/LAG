"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-11 17:51:46
FilePath     : /LAG/envs/JSBSim/termination_conditions/safe_return.py
Description  : 
"""

from .termination_condition_base import BaseTerminationCondition


class SafeReturn(BaseTerminationCondition):
    """End up the simulation if :
        1. the current aircraft has been shot down
        2. all the enemy aircraft has been destroyed while current aircraft is not under attack

    Args:
        BaseTerminationCondition (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.

        End up the simulation if:
            - the current aircraft has been shot down.
            - all the enemy aircraft has been destroyed while current aircraft is not under attack.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        # the current aircraft has crashed
        if env.agents[agent_id].is_shot_down:
            self.log(f"{agent_id} has been shot down! Total Steps = {env.current_step}")
            return True, False, info

        elif env.agents[agent_id].is_crash:
            self.log(f"{agent_id} has crashed! Total Steps = {env.current_step}")
            return True, False, info

        # all the enemy aircraft has been destroyed while current aircraft is not under attack
        elif all(
            [not enemy.is_alive for enemy in env.agents[agent_id].enemies]
        ) and all(
            [not missile.is_alive for missile in env.agents[agent_id].under_missiles]
        ):
            self.log(f"{agent_id} mission completed! Total Steps = {env.current_step}")
            return True, True, info

        else:
            return False, False, info
