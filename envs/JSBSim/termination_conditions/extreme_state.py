"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-11 17:48:54
FilePath     : /LAG/envs/JSBSim/termination_conditions/extreme_state.py
Description  : 
"""
from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = bool(env.agents[agent_id].get_property_value(c.detect_extreme_state))
        if done:
            env.agents[agent_id].crash()
            self.log(
                f"{agent_id} is on an extreme state! Total Steps={env.current_step}"
            )
        success = False
        return done, success, info
