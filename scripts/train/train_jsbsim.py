#!/usr/bin/env python
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from config import get_config, parser_to_dict, parser_dict_to_color_string
from logger import init_logger, set_color
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    ShareSubprocVecEnv,
    ShareDummyVecEnv,
)


def make_train_env(all_args):
    """Create training env based on the provided arguments.

    Args:
        all_args (namespace): A namespace containing environment configuration arguments.
        Which must include 'env_name', 'scenario_name', 'seed' and 'n_rollout_threads'.

    Returns:
        A vectorized environment instance, either a DummyVecEnv or SubprocVecEnv,
        depending on the number of rollout threads specified in all_args.

    This function initializes environments for training based on the specified
    environment name ('SingleCombat', 'SingleControl', or 'MultipleCombat').
    It sets up the environment creation functions, seeds each environment for
    reproducibility, and returns a vectorized environment for parallel processing.
    """

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error(
                    "Can not support the " + all_args.env_name + "environment."
                )
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )


def make_eval_env(all_args):
    """Create evaluation environments based on the provided arguments.

    Args:
        all_args: A namespace containing environment configuration arguments.
                    Must include 'env_name', 'scenario_name', 'seed', and 'n_eval_rollout_threads'.

    Returns:
        A vectorized environment instance, either a DummyVecEnv or SubprocVecEnv,
        depending on the number of evaluation rollout threads specified in all_args.

    This function initializes environments for evaluation based on the specified
    environment name ('SingleCombat', 'SingleControl', or 'MultipleCombat').
    It sets up the environment creation functions, seeds each environment for
    reproducibility with a unique seed per rank, and returns a vectorized environment
    for parallel processing.

    The seeding scheme is designed to ensure distinct seeds for each environment,
    avoiding accidental correlations between multiple random number generators.
    """

    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error(
                    "Can not support the " + all_args.env_name + "environment."
                )
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env

        return init_env

    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
            )
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]
            )


def parse_args(args, parser):
    """Parse command line arguments for JSBSim environment parameters.

    Args:
        args (list): List of command line arguments.
        parser (argparse.ArgumentParser): Argument parser instance.

    Returns:
        argparse.Namespace: Parsed command line arguments encapsulated in a namespace.

    This function adds a group of arguments related to JSBSim environment parameters to the parser.
    It includes the scenario name which determines which scenario to run on.
    The parsed arguments are then returned as an argparse.Namespace object.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> args = ["--scenario-name", "custom_scenario"]
        >>> parsed_args = parse_args(args, parser)
        >>> print(parsed_args.scenario_name)
        custom_scenario
    """
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument(
        "--scenario-name",
        type=str,
        default="singlecombat_simple",
        help="Which scenario to run on",
    )
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    parser_dict = parser_to_dict(parser)
    all_args_info = parser_dict_to_color_string(parser_dict)
    all_args = parse_args(args, parser)

    # logger
    init_logger(all_args)
    logger = logging.getLogger()
    logger.info(all_args_info)

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logger.critical("Device: GPU\n")
        # use cuda mask to control using which GPU
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logger.critical("Device: CPU\n")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = (
        Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(
            job_type="training",
            dir=str(run_dir),
            config=all_args,
            project=all_args.env_name,
            reinit=True,
            group=all_args.scenario_name,
            name=f"{all_args.experiment_name}_seed{all_args.seed}",
            notes=socket.gethostname(),
        )
    else:
        if not run_dir.exists():
            curr_run = "run_1"
        else:
            exist_run_nums = [
                int(str(folder.name).split("run")[1])
                for folder in run_dir.iterdir()
                if str(folder.name).startswith("run")
            ]
            if len(exist_run_nums) == 0:
                curr_run = "run_1"
            else:
                curr_run = "run_%i" % (max(exist_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # env init
    train_envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": train_envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    try:
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        train_envs.close()
        if all_args.use_wandb:
            run.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
