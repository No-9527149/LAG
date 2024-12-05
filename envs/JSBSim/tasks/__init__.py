"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:07:45
LastEditTime : 2024-11-27 20:16:52
FilePath     : /LAG/envs/JSBSim/tasks/__init__.py
Description  : 
"""

from .heading_task import HeadingTask
from .single_combat_task import SingleCombatTask, HierarchicalSingleCombatTask
from .single_combat_with_missile_task import (
    SingleCombatDodgeMissileTask,
    HierarchicalSingleCombatDodgeMissileTask,
    HierarchicalSingleCombatShootMissileTask,
    SingleCombatShootMissileTask,
)
from .collaboration_task import CollaborationTask, HierarchicalCollaborationTask
