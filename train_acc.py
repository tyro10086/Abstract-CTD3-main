import os, sys
import tqdm
import math
import time
import logging

import numpy as np
from argparse import ArgumentParser

import gym
import highway_env

import torch
from tensorboardX import SummaryWriter

import utils

from algo import *
import time

from algo.td3_risk_disturbance import DotProductSimilarity


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--project", type=str, help="the path to save project",
                        default="project/20231026-TD3_risk-acc")
    parser.add_argument("--model_config", type=str, help="path of rl algorithm configuration",

                        default="conf/algorithm/TD3_risk_acc.yaml")

    parser.add_argument("--env_config", type=str, help="path of highway env",
                        default="conf/env/highway_acc_continuous_acceleration.yaml")
    parser.add_argument("--gpu", type=str, help="[0,1,2,3 | -1] the id of gpu to train/test, -1 means using cpu",
                        default="0")

    args = parser.parse_args()
    return args


def build_env(env_name, configure):
    """build the highway environment"""

    env = gym.make(env_name, render_mode='human')
    env.configure(configure)
    env.reset()

    return env


def build_agent(config, gpu, writer):
    """choose reinforcement learning algorithm"""
    if config["algo_name"] == "TD3":
        agent = TD3(config, gpu, writer=writer)
    elif config["algo_name"] == "TD3_risk_disturbance":
        agent = TD3_risk_disturbance(config, gpu, writer=writer)

    return agent


def get_reward(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your reward function here! """

    reward = 0.

    ego_presence, ego_x, ego_vx = state[0]
    lead_presence, lead_relative_x, lead_relative_vx = state[1]

    reward += (0.05 * ego_vx)

    return reward


def get_cost(info, state, action, done, truncated, last_action=None, last_state=None):
    """ define your risk function here! """
    cost = 0
    if info["crashed"]:
        cost = 100

    return cost


def eval_policy(agent, eval_env, env_config, algo_config, eval_episodes=10, seed=1024):
    """evalurate the agent during training"""
    avg_reward = 0.
    num_crash = 0

    for _ in range(eval_episodes):
        state = eval_env.reset()

        state = utils.normalize_observation(state,
                                            env_config["config"]["observation"]["features"],
                                            env_config["config"]["observation"]["features_range"],
                                            clip=False)
        state = state.flatten()
        state = state[-2:]
        # print(state)

        done, truncated = False, False
        while not done and not truncated:

            action = agent.choose_action(np.array(state))

            if algo_config["model"]["action_dim"] == 1:
                real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
                real_action = np.array(real_action).clip(eval_env.action_space.low, eval_env.action_space.high)

            state, reward, done, truncated, info = eval_env.step(real_action)

            if info["crashed"]:
                num_crash += 1

            reward = get_reward(info, state, action, done, truncated)
            avg_reward += reward

            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]

    avg_reward /= eval_episodes

    return avg_reward, num_crash


def train(env, eval_env, agent, max_timesteps, max_updates, print_freq, save_freq, env_config, algo_config, project_dir,
          writer):
    """train the agent"""
    # episode周期碰撞次数
    episode_crash_list = list()
    # episode累计奖励
    episode_reward_list = list()
    # 每个episode的步长
    episode_step_list = list()

    trajectory_list = list()

    average_time_per_step = 0.
    num_steps = 0

    ave_reward_list = list()
    num_crash_list = list()
    eval_step_list = list()

    state = env.reset(seed=1024)

    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    # 相似度
    relative = DotProductSimilarity()

    for t_step in range(max_timesteps):
        t_step += 1
        episode_timesteps += 1

        if algo_config["render"]: env.render()

        """Select action according to policy"""
        beg_time = time.time()
        action = agent.choose_action(state)
        end_time = time.time()
        average_time_per_step += (end_time - beg_time)
        num_steps += 1

        if "exploration_noise" in algo_config["trainer"]:
            noise = np.random.normal(0, algo_config["trainer"]["exploration_noise"],
                                     size=algo_config["model"]["action_dim"])
            noise = noise.clip(-algo_config["trainer"]["noise_clip"], algo_config["trainer"]["noise_clip"])
            action = action + noise

        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]

        real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        """Perform action"""
        next_state, reward, done, truncated, info = env.step(real_action)

        """Define the reward composition"""
        reward = get_reward(info, next_state, action, done, truncated)

        """Define the cost"""
        cost = get_cost(info, next_state, action, done, truncated)
        # print("cost:", cost)
        is_crash = 1 if info["crashed"] else 0

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        if "risk" in algo_config["algo_name"]:
            agent.memory.push(state, action, reward, next_state, done, cost)
            logging.info('%s, %s, %s, %s, %s, %s, %s', episode_num, state, action, reward, next_state, done, cost)
        else:
            agent.memory.push(state, action, reward, next_state, done)
            logging.info('%s, %s, %s, %s, %s, %s', episode_num, state, action, reward, next_state, done)

        # 危险场景库
        if cost == 100:
            flag = False
            for data in agent.danger_scenario.buffer:
                fal_state, fal_action, fal_reward, fal_next_state, is_done, exist_cost = data
                state_tmp = torch.tensor(state.reshape(1, -1)).float()
                fal_state_tmp = torch.tensor(fal_state.reshape(1, -1)).float()
                action_tmp = torch.tensor(action.reshape(1, -1)).float()
                fal_action_tmp = torch.tensor(fal_action.reshape(1, -1)).float()
                if relative(state_tmp, fal_state_tmp) > 0.75 and relative(action_tmp, fal_action_tmp) > 0.75:
                    flag = True
                    break
            if flag is False:
                if "risk" in algo_config["algo_name"]:
                    agent.danger_scenario.push(state, action, reward, next_state, done, cost)
                else:
                    agent.danger_scenario.push(state, action, reward, next_state, done)

        """Train agent"""
        agent.update(num_iteration=max_updates)
        state = next_state

        """some information"""
        episode_reward += reward
        env.render()

        # done表示撞车 truncated步长结束
        if done or truncated:
            episode_num += 1

            episode_crash_list.append(is_crash)
            episode_reward_list.append(episode_reward)
            episode_step_list.append(t_step)

            writer.add_scalar('Reward/epoch_reward', episode_reward, global_step=episode_num)
            writer.add_scalar('crashs/count', is_crash, global_step=episode_num)

            now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            # print(f"[{now_time}][Train] Step: {t_step}, Episode: {episode_num},",
            #       f"Episode Steps: {episode_timesteps}, Reward: {episode_reward:.2f}",
            #       f"crash Num: {sum(episode_crash_list)}")
            logging.info('发生撞车：Step:%s, Episode: %s, Episode Step：%s, Reward：%s, crash Num:%s', t_step, episode_num, episode_timesteps, episode_reward,
                         sum(episode_crash_list))

            episode_reward = 0
            episode_timesteps = 0
            is_crash = 0

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]

        if t_step % algo_config["trainer"]["eval_freq"] == 0:
            with torch.no_grad():
                eval_episodes = algo_config["trainer"]["eval_episodes"]
                ave_reward, num_crash = eval_policy(agent, eval_env, env_config, algo_config,
                                                    eval_episodes=eval_episodes)

                ave_reward_list.append(ave_reward)
                num_crash_list.append(num_crash)
                eval_step_list.append(t_step)
                # now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                # print(f"[{now_time}][Eval] Step: {t_step}, Episode Num: {eval_episodes},",
                #       f"Ave Reward: {ave_reward:.2f}",
                #       f"crash Num: {num_crash}")
                logging.info('Step:%s, Episode Num:%s, Ave Reward:%s, crash Num:%s', t_step, eval_episodes, ave_reward, num_crash)

    save_dir = os.path.join(project_dir, "checkpoint")
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)

    # print(f"Average decision time per step: {average_time_per_step / num_steps} s")
    logging.info('Average decision time per step: %s', average_time_per_step / num_steps)

    return episode_reward_list, episode_crash_list, episode_step_list


def main():
    config = parse_args()
    utils.make_dir(config.project, config.model_config, config.env_config)

    tensorboard_dir = os.path.join(config.project, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    env_config = utils.load_yml(config.env_config)
    env = build_env(env_config["env_name"], env_config["config"])

    eval_env = build_env(env_config["env_name"], env_config["config"])

    algo_config = utils.load_yml(config.model_config)
    agent = build_agent(algo_config, config.gpu, writer)

    episode_reward_list, episode_crash_list, episode_step_list = train(env, eval_env, agent,
                                                                       algo_config["trainer"]["max_timesteps"],
                                                                       algo_config["trainer"]["max_updates"],
                                                                       algo_config["trainer"]["print_freq"],
                                                                       algo_config["trainer"]["save_freq"], env_config,
                                                                       algo_config, config.project, writer)

    np.save(os.path.join(config.project, "results", "train_epoch_return.npy"), episode_reward_list)
    np.save(os.path.join(config.project, "results", "train_epoch_crash.npy"), episode_crash_list)
    np.save(os.path.join(config.project, "results", "train_epoch_step.npy"), episode_step_list)

    utils.single_plot(episode_step_list, episode_reward_list,
                      path=os.path.join(config.project, "plots", "train_step_return"),
                      xlabel="Step", ylabel="Return", title="", label=algo_config["algo_name"])
    utils.single_plot(None, episode_reward_list,
                      path=os.path.join(config.project, "plots", "train_epoch_return"),
                      xlabel="Epoch", ylabel="Return", title="", label=algo_config["algo_name"])

    utils.single_plot(episode_step_list, episode_crash_list,
                      path=os.path.join(config.project, "plots", "train_step_crash"),
                      xlabel="Step", ylabel="Num Of crash", title="", label=algo_config["algo_name"])
    utils.single_plot(None, episode_crash_list,
                      path=os.path.join(config.project, "plots", "train_epoch_crash"),
                      xlabel="Epoch", ylabel="Num Of crash", title="", label=algo_config["algo_name"])

    env.close()
    eval_env.close()


if __name__ == '__main__':
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        # format='%(asctime)s [%(levelname)s] %(message)s',
        format='%(message)s',
        handlers=[
            logging.FileHandler('trained_acc_data.log'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ],
        filemode='w'
    )
    logging.info('episode rel_dis rel_speed acc reward next_rel_dis next_rel_speed done cost')
    main()
