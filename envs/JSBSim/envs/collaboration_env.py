"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-27 20:10:23
LastEditTime : 2024-12-05 15:52:56
FilePath     : /LAG/envs/JSBSim/envs/collaboration_env.py
Description  : 
"""
import numpy as np
from .env_base import BaseEnv
from ..tasks import CollaborationTask, HierarchicalCollaborationTask


class CollaborationEnv(BaseEnv):
    """For collaboration environment.

    Args:
        BaseEnv (_type_): _description_
    """
    def __init__(self, env_config_name: str):
        super().__init__((env_config_name))
        # TODO(zzp): Env specific initialization
        

    def load_task(self):
        task_name = getattr(self.env_config, "task", "None")
        if task_name == "collaboration_task":
            self.task = CollaborationTask(self.env_config)
        elif task_name == "hierarchical_collaboration_task":
            self.task == HierarchicalCollaborationTask(self.env_config)
        else:
            raise NotImplementedError(f"Unknown task name: {task_name}")

    def get_initial_states(self):
        r = self.task.initial_distance
        alt = self.task.initial_alt
        random_theta = np.random.uniform(np.pi / 3, np.pi / 2)
        random_phi = np.random.uniform(np.pi / 3, 2 * np.pi / 3)
        x = r * np.sin(random_theta) * np.cos(random_phi)
        y = r * np.sin(random_theta) * np.sin(random_phi)
        z = r * np.cos(random_theta) + alt
        
        
        

    def reset(self) -> np.ndarray:
        # TODO(zzp): rewrite env reset here
        self.current_step = 0
        self.reset_simulators()
        # NOTE(zzp): task.reset(self), watch the self here
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # NOTE(zzp): initial_state is a dict
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        # ? _temp_sims here for what?
        self._temp_sims.clear()
