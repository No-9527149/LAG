"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 16:17:21
LastEditTime : 2024-11-11 16:19:36
FilePath     : /LAG/logger/utils.py
Description  : 
"""

import datetime
import os
import numpy as np


def get_local_time():
    """Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H:%M:%S")
    return cur


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
