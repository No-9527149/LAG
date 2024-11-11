"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-11 17:51:21
FilePath     : /LAG/envs/JSBSim/termination_conditions/timeout.py
Description  : 
"""
from .termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """Episode terminates if max_step steps has passed.

    Args:
        BaseTerminationCondition (_type_): _description_
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_steps = getattr(self.config, "max_steps", 500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = env.current_step >= self.max_steps
        if done:
            self.log(f"{agent_id} step limits! Total Steps = {env.current_step}")
        success = False
        return done, success, info
