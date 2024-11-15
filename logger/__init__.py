'''
Author       : zzp@buaa.edu.cn
Date         : 2024-11-11 20:47:48
LastEditTime : 2024-11-11 21:31:14
FilePath     : /LAG/logger/__init__.py
Description  : No more description
'''

import logging
import os
import colorlog
import re
from colorama import init
from logger.utils import get_local_time, ensure_dir
from pathlib import Path

log_colors_config = {
    "DEBUG": "green",
    "WARNING": "yellow",
    "ERROR": "pink",
    "CRITICAL": "red",
}


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            record.msg = ansi_escape.sub("", str(record.msg))
        return True


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"


def init_logger(all_args):
    """A logger that can show a message on standard output and write it into the file
    named `filename` simultaneously. All the message that you want to log MUST be str

    Args:
        config (Config): An instance object of Config, used to record parameter info

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = "./log/"
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)
    log_dir_name = (Path(dir_name) / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name)
    ensure_dir(log_dir_name)
    log_file_name = "{}.log".format(get_local_time())
    log_file_path = os.path.join(log_dir_name, log_file_name)

    file_fmt = "%(asctime)-15s %(levelname)s  %(message)s"
    file_date_fmt = "%a %d %b %Y %H:%M:%S"
    file_formatter = logging.Formatter(file_fmt, file_date_fmt)

    str_fmt = "%(log_color)s%(asctime)-15s %(levelname)s %(message)s"
    str_date_fmt = "%d %b %H:%M"
    str_formatter = colorlog.ColoredFormatter(
        str_fmt, str_date_fmt, log_colors=log_colors_config
    )

    if all_args.log_state is None or all_args.log_state.lower() == "info":
        level = logging.INFO
    elif all_args.log_state.lower() == "debug":
        level = logging.DEBUG
    elif all_args.log_state.lower() == "error":
        level = logging.ERROR
    elif all_args.log_state.lower() == "warning":
        level = logging.WARNING
    elif all_args.log_state.lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(level)
    fh.setFormatter(file_formatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(str_formatter)

    logging.basicConfig(level=level, handlers=[sh, fh])