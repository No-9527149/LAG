"""
Author       : zzp@buaa.edu.cn
Date         : 2024-11-18 16:40:15
LastEditTime : 2024-11-27 15:17:09
FilePath     : /LAG/runner/collaboration_runner.py
Description  : 
"""

import sys
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List
from logger import set_color
from .base_runner import Runner, ReplayBuffer
from .jsbsim_runner import JSBSimRunner


def _t2n(x):
    return x.detach().cpu().numpy()


class CollaborationRunner(JSBSimRunner):

    def load(self):
        # load parameters
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.envs.use_selfplay
        assert self.use_selfplay == False, "Collaboration can not use selfplay"

        # policy and algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplemented
        self.policy = Policy(
            self.all_args, self.obs_space, self.act_space, device=self.device
        )
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(
            self.all_args, self.num_agents, self.obs_space, self.act_space
        )

        if self.model_dir is not None:
            self.restore()

    def run(self):
        self.warmup()
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in range(episodes):
            logging.info(
                set_color(">>>>>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<", "red")
            )
            heading_turns_list = []
            for step in tqdm(
                range(self.buffer_size),
                desc=set_color(f"Collect {episode:>4}", "yellow"),
            ):
                # sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states_actor,
                    rnn_states_critic,
                ) = self.collect(step)
                # observe reward and next observation
                obs, rewards, dones, infos = self.envs.step(actions)
                # extra recorded information
                # TODO(zzp): info
                for info in infos:
                    if "heading_turn_counts" in info:
                        heading_turns_list.append(info["heading_turn_counts"])
                # generate insert data
                data = (
                    obs,
                    actions,
                    rewards,
                    dones,
                    action_log_probs,
                    values,
                    rnn_states_actor,
                    rnn_states_critic,
                )
                # insert data into buffer
                self.insert(data)
            # compute return and update network
            self.compute()
            train_infos = self.train()
            # post process
            self.total_num_steps = (
                (episode + 1) * self.buffer_size * self.n_rollout_threads
            )
            # log info
            if episode % self.log_interval == 0:
                end = time.time()
                # NOTE(zzp): FPS means number of steps per sec
                logging.info(
                    set_color("Episode: ", "green")
                    + set_color("{} / {}".format(episode, episodes), "blue")
                    + set_color("Timestep: ", "green")
                    + set_color(
                        "{} / {}".format(self.total_num_steps, self.num_env_steps),
                        "blue",
                    )
                    + set_color("FPS: ", "green")
                    + set_color("{}".format(int(self.total_num_steps / (end - start))))
                )
                train_infos["average_episode_rewards"] = (
                    self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                )
                logging.info(
                    set_color("Reward : ", "cyan")
                    + "{}".format(train_infos["average_episode_rewards"]),
                )
                if len(heading_turns_list):
                    train_infos["average_episode_rewards"] = np.mean(heading_turns_list)
                    logging.info(
                        set_color("Turn   : ", "pink")
                        + "{}".format(train_infos["average_episode_turns"])
                    )
                self.log_info(train_infos, self.total_num_steps)
            logging.info(
                set_color(
                    ">>>>>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<", "pink"
                )
            )
            # eval
            if self.use_eval and episode != 0 and episode % self.eval_interval == 0:
                self.eval(self.total_num_steps)
            # sava model
            if episode % self.save_interval == 0 or episode == episode - 1:
                self.save(episode)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = (
            self.policy.get_actions(
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states_actor[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        # split parallel data [N * M, shape] -> [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        (
            obs,
            actions,
            rewards,
            dones,
            action_log_probs,
            values,
            rnn_states_actor,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        self.buffer.insert(
            obs,
            actions,
            rewards,
            masks,
            action_log_probs,
            values,
            rnn_states_actor,
            rnn_states_critic,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info(set_color(">>>>>>>>>>>>>>>>>>>>>>Evaluate<<<<<<<<<<<<<<<<<<<<<<", "red"))
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, * self.buffer.rewards.shape[2:]), dtype=np.float32)
        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, * self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        while total_episodes < self.eval_episodes:
            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            # observe reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0
            eval_masks = np.zeros_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)
        eval_infos = {}
        eval_infos["eval_average_episode_rewards"] = np.concatenate(eval_episode_rewards).mean(axis=1)
        logging.info(set_color("Reward", "green") + "{}".format(np.mean(eval_infos["eval_average_episode_rewards"])))
        self.log_info(eval_infos, total_num_steps)
        logging.info(set_color(">>>>>>>>>>>>>>>>>>>>>Evaluate<<<<<<<<<<<<<<<<<<<<<", "pink"))

    def save(self, episode):
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(self.save_dir) + "/actor_latest.pt")
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(self.save_dir) + "/critic_latest.pt")

    # TODO(zzp): render
