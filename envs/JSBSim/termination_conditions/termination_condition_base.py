"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-11 17:49:26
FilePath     : /LAG/envs/JSBSim/termination_conditions/termination_condition_base.py
Description  : 
"""
import logging
from abc import ABC, abstractmethod


class BaseTerminationCondition(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env, agent_id, info={}):
        """Return whether the episode should terminate.
        Overwrite by subclass

        Args:
            task (_type_): task instance
            env (_type_): env instance
            agent_id (_type_): _description_
            info (dict, optional): _description_. Defaults to {}.

        Returns:
            (tuple): (done, success, info)
        """
        raise NotImplementedError

    def log(self, msg):
        logging.debug(msg)
