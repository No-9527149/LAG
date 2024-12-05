"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-27 20:17:49
FilePath     : /LAG/envs/JSBSim/envs/single_combat_env.py
Description  : 
"""

import numpy as np
from .env_base import BaseEnv
from ..tasks import (
    SingleCombatTask,
    SingleCombatDodgeMissileTask,
    HierarchicalSingleCombatDodgeMissileTask,
    HierarchicalSingleCombatShootMissileTask,
    SingleCombatShootMissileTask,
    HierarchicalSingleCombatTask,
)


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """

    def __init__(self, env_config_name: str):
        super().__init__(env_config_name)
        # Env-Specific initialization here!
        assert (
            len(self.agents.keys()) == 2
        ), f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        task_name = getattr(self.env_config, "task", None)
        if task_name == "single_combat":
            self.task = SingleCombatTask(self.env_config)
        elif task_name == "hierarchical_single_combat":
            self.task = HierarchicalSingleCombatTask(self.env_config)
        elif task_name == "single_combat_dodge_missile":
            self.task = SingleCombatDodgeMissileTask(self.env_config)
        elif task_name == "single_combat_shoot":
            self.task = SingleCombatShootMissileTask(self.env_config)
        elif task_name == "hierarchical_single_combat_dodge_missile":
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.env_config)
        elif task_name == "hierarchical_single_combat_shoot":
            self.task = HierarchicalSingleCombatShootMissileTask(self.env_config)
        else:
            raise NotImplementedError(f"Unknown task name: {task_name}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        # switch side
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        # TODO(zzp): some comments in original codes
        # self.init_states[0].update({
        #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
        #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
        # })
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(init_states[idx])
        self._temp_sims.clear()
