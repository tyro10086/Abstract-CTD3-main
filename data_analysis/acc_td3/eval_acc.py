import logging
import os
import copy
from tqdm import *

import gym
import joblib
import numpy as np

import highway_env
import utils
from algo import *

def eval_trad(env, agent, env_config, algo_config, mdl):
    #   初始化状态
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    #   确定步数
    max_timesteps = algo_config["trainer"]["max_timesteps"]
    max_timesteps = max_timesteps // 10

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
        #   获取簇标签，并得到聚类中心的状态
        label = mdl.predict(np.array(state, dtype='float').reshape(1, -1))
        state_predict = mdl.cluster_centers_[label]

        #   利用聚类中心状态选择动作并处理格式
        action_predict = agent.choose_action(state_predict)
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict = [action_predict[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict = np.array(real_action_predict).clip(env.action_space.low, env.action_space.high)

        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)
        next_state_predict, _, done, truncated, info = env_predict.step(real_action_predict)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        next_state_predict = utils.normalize_observation(next_state_predict,
                                                    env_config["config"]["observation"]["features"],
                                                    env_config["config"]["observation"]["features_range"],
                                                    clip=False)
        next_state_predict = next_state_predict.flatten()
        next_state_predict = next_state_predict[-2:]

        logging.info('%s, %s, %s %s', episode_num, next_state, next_state_predict, label)

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]

def eval_new(env, agent,  env_config, algo_config, eval_config, mdl):
    #   初始化状态
    state = env.reset(seed=1024)
    state = utils.normalize_observation(state,
                                        env_config["config"]["observation"]["features"],
                                        env_config["config"]["observation"]["features_range"],
                                        clip=False)
    state = state.flatten()
    state = state[-2:]

    episode_num = 1

    #   确定步数
    max_timesteps = algo_config["trainer"]["max_timesteps"]
    max_timesteps = max_timesteps // 10

    for t_step in tqdm(range(max_timesteps)):
        t_step += 1

        #   选择动作并处理格式
        action = agent.choose_action(np.array(state))
        if algo_config["model"]["action_dim"] == 1:
            real_action = [action[0], algo_config["model"]["action_config"]["steering"]]
            real_action = np.array(real_action).clip(env.action_space.low, env.action_space.high)

        #   深拷贝当前环境，用于预测值的env.step
        env_predict = copy.deepcopy(env)
        #   获取簇标签，并得到聚类中心的状态

        state = np.array(state, dtype='float').reshape(1, -1)

        #   state区间化，输入聚类模型找中心点
        state_ = utils.intervalize_state(state, eval_config)[0]
        label = mdl.predict(state_)[0]
        state_predict = mdl.cluster_centers_[label]

        #   利用聚类中心状态选择动作并处理格式
        state_innet = np.concatenate(((state_predict[:, 0] + state_predict[:, 1]) / 2,
                                     (state_predict[:, 2] + state_predict[:, 3]) / 2), axis=0)

        action_predict = agent.choose_action(state_innet)
        if algo_config["model"]["action_dim"] == 1:
            real_action_predict = [action_predict[0], algo_config["model"]["action_config"]["steering"]]
            real_action_predict = np.array(real_action_predict).clip(env.action_space.low, env.action_space.high)

        """Perform action"""
        next_state, _, done, truncated, info = env.step(real_action)
        next_state_predict, _, done, truncated, info = env_predict.step(real_action_predict)

        """Store data in replay buffer"""
        next_state = utils.normalize_observation(next_state,
                                                 env_config["config"]["observation"]["features"],
                                                 env_config["config"]["observation"]["features_range"],
                                                 clip=False)
        next_state = next_state.flatten()
        next_state = next_state[-2:]

        next_state_predict = utils.normalize_observation(next_state_predict,
                                                    env_config["config"]["observation"]["features"],
                                                    env_config["config"]["observation"]["features_range"],
                                                    clip=False)
        next_state_predict = next_state_predict.flatten()
        next_state_predict = next_state_predict[-2:]

        logging.info('%s, %s, %s %s', episode_num, next_state, next_state_predict, label)

        state = next_state

        if done or truncated:
            episode_num += 1

            state = env.reset(seed=1024)
            state = utils.normalize_observation(state,
                                                env_config["config"]["observation"]["features"],
                                                env_config["config"]["observation"]["features_range"],
                                                clip=False)
            state = state.flatten()
            state = state[-2:]

def eval_acc(config):
    #   创建日志，要求可重写
    logging.basicConfig(
        filename='./../../test.log',
        #filename='./../../trained_acc_data.log',
        level=logging.INFO,
        filemode='w',
        # format='%(asctime)s [%(levelname)s] %(message)s',
        format='%(message)s',
        # handlers=[
        #     logging.FileHandler('./../../trained_acc_data.log'),  # 输出到文件
        #     logging.StreamHandler()  # 输出到控制台
        # ]
    )
    logger = logging.getLogger('log')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    #   格式时epi, 状态真实值, 状态预测值, 簇标签
    logging.info('Episode rel_dis_true rel_speed_true rel_dis_predict rel_speed_predict label')

    #   创建环境
    path = os.path.join("./../../", config.env_config)
    env_config = utils.load_yml(path)
    env = gym.make(env_config["env_name"], render_mode='human')
    env.configure(env_config["config"])

    #   创建agent
    path = os.path.join("./../../", config.model_config)
    algo_config = utils.load_yml(path)
    if algo_config["algo_name"] == "TD3":
        agent = TD3(algo_config, config.gpu, writer=None)
    elif algo_config["algo_name"] == "TD3_risk_disturbance":
        agent = TD3_risk_disturbance(algo_config, config.gpu, writer=None)

    #   加载参数
    agent.load(os.path.join("./../../", config.checkpoint))

    save_mdl_path = "./mdls/"
    if not config.algo:
        if config.mode == 'gap':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_gap.pkl')
        elif config.mode == 'canopy':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_canopy.pkl')
        elif config.mode == 'elbow':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_elbow.pkl')
        elif config.mode == 'silhouette':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_silhouette.pkl')
        else:
            print("ERROR: check the value of parameter mode")
            exit(0)
        eval_trad(env, agent, env_config, algo_config, mdl)

    elif config.algo:
        if config.mode == 'gap':
            mdl = joblib.load(save_mdl_path + 'new_kmeans_gap.pkl')
        elif config.mode == 'canopy':
            mdl = joblib.load(save_mdl_path + 'new_kmeans_canopy.pkl')
        elif config.mode == 'elbow':
            mdl = joblib.load(save_mdl_path + 'new_kmeans_elbow.pkl')
        elif config.mode == 'silhouette':
            mdl = joblib.load(save_mdl_path + 'new_kmeans_silhouette.pkl')
        else:
            print("ERROR: check the value of parameter mode")
            exit(0)
        path = os.path.join("./../../", config.eval_config)
        eval_config = utils.load_yml(path)
        eval_new(env, agent, env_config, algo_config, eval_config, mdl)