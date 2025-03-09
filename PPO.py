import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import pickle as pkl
from tqdm import *
import yaml
import torch
from models.PPO_model import PPO
from config import config_mario, config_mnist
from utils.env import Mario_env, Mnist_env
from utils.utils import set_seed

if __name__ == '__main__':

    # =================== setup ===============================
    parser = argparse.ArgumentParser(description='Metarule Learner')
    # =================== basic setup =========================
    parser.add_argument('--dataset', type=str,
                        help='mario or mnist', default='mnist')
    args = parser.parse_args()
    args = {**vars(args)}
    if args['dataset'] == 'mario':
        config_object = config_mario.config
    if args['dataset'] == 'mnist':
        config_object = config_mnist.config
    set_seed(config_object['seed'])

    # ==================== parameter setup ====================
    num_episodes = config_object['num_episodes']
    actor_lr = config_object['actor_lr']
    eps = config_object['eps']
    n_hiddens = config_object['n_hiddens']
    epochs= config_object['epochs']
    num_trails_per_episode = config_object['num_trails_per_episode']

    # ==================== path setup =========================
    model_path = config_object['model_path']
    pl_path = config_object['pl_path']
    result_path = config_object['result_path']
    excel_file_path = config_object['excel_result_path']
    log_file_path = config_object['log_file_path']

    # ==================== task setup =========================
    if config_object['dataset'] == 'mario':
        train_task = config_object['train_task']
        test_task = config_object['test_task']
    elif config_object['dataset'] == 'mnist':
        all_task = config_object['all_task']

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # if os.path.exists(log_file_path):
    #     os.remove(log_file_path)

    # ==================== save_config =======================
    with open(result_path+'config.txt', 'w') as f:
        yaml.dump(config_object, f)

    # ==================== device setup =======================
    device = torch.device("cuda:%s" % config_object['GPU'] if torch.cuda.is_available() else "cpu")
    config_object['device'] = device

    # ==================== env setup =========================
    if config_object['dataset'] == 'mario':
        env = Mario_env(config_object)
    elif config_object['dataset'] == 'mnist':
        env = Mnist_env(config_object)

    # ==================== build model ========================
    state_shape = env.get_state_shape()
    agent = PPO(state_shape, config_object)

    # ==================== on policy training =================
    train_reward_list = []
    test_in_train_case_reward_list = []
    test_case_reward_list = []
    if config_object['dataset'] == 'mario':
        test_task_reward_list = []

    for i in tqdm(range(num_episodes)):
        state = env.reset()
        episode_train_reward = []
        episode_test_in_train_case_reward = []
        episode_test_case_reward = []
        if config_object['dataset'] == 'mario':
            episode_test_task_reward = []

        # save data in this episode
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
        }

        for j in range(num_trails_per_episode):
            action = agent.take_action(np.expand_dims(state,axis=0), 'train')
            reward = env.step(action)

            # save states\actions...every timestep
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)

            # new state
            state = env.reset()
            # reward sum
            episode_train_reward.append(np.mean(reward))

        # train
        transition_dict['states'] = np.array(transition_dict['states'])
        transition_dict['actions'] = np.array(transition_dict['actions'])
        transition_dict['rewards'] = np.array(transition_dict['rewards'])

        agent.learn(transition_dict)

        # save the reward of this episode
        train_reward_list.append(np.mean(episode_train_reward))

        # test in train case to get a reward that can measure without exploratory disturbance
        if config_object['dataset'] == 'mario':
            for task_name in train_task:
                state = env.reset(task_name)
                action = agent.take_action(np.expand_dims(state, axis=0), 'test')
                reward = env.step(action)
                episode_test_in_train_case_reward.append(reward)
            test_in_train_case_reward_list.append(np.mean(episode_test_in_train_case_reward))


            # test case in train tasks
            for task_name in train_task:
                state = env.reset_test(task_name)
                action = agent.take_action(np.expand_dims(state, axis=0), 'test')
                reward = env.step(action)
                episode_test_case_reward.append(reward)
            test_case_reward_list.append(np.mean(episode_test_case_reward))

            # test tasks
            for task_name in test_task:
                state = env.reset_test(task_name)
                action = agent.take_action(np.expand_dims(state, axis=0), 'test')
                reward = env.step(action)
                episode_test_task_reward.append(reward)
            test_task_reward_list.append(np.mean(episode_test_task_reward))

        elif config_object['dataset'] == 'mnist':
            for task_name in all_task:
                state = env.reset(task_name)
                action = agent.take_action(np.expand_dims(state, axis=0), 'test')
                reward = env.step(action)
                episode_test_in_train_case_reward.append(reward)
            test_in_train_case_reward_list.append(np.mean(episode_test_in_train_case_reward))

            # test case in train tasks
            for task_name in all_task:
                state = env.reset_test(task_name)
                action = agent.take_action(np.expand_dims(state, axis=0), 'test')
                reward = env.step(action)
                episode_test_case_reward.append(reward)
            test_case_reward_list.append(np.mean(episode_test_case_reward))

        with open(log_file_path, 'a') as f:
            print(f'iter:{i}', file=f)
            print(f'iter:{i}')
            print(f'train reward:{train_reward_list[-1:]}', file=f)
            print(f'train reward:{train_reward_list[-1:]}')
            print(f'test in train case reward:{test_in_train_case_reward_list[-1:]}', file=f)
            print(f'test in train case reward:{test_in_train_case_reward_list[-1:]}')
            print(f'test case reward:{test_case_reward_list[-1:]}', file=f)
            print(f'test case reward:{test_case_reward_list[-1:]}')
            if config_object['dataset']=='mario':
                print(f'test task reward:{test_task_reward_list[-1:]}', file=f)
                print(f'test task reward:{test_task_reward_list[-1:]}')

        # save model and plot
        if i % 25 == 0:
            torch.save(agent, model_path + 'agent_iter'+str(i)+'.pt')

            plt.plot(train_reward_list)
            plt.title('train reward')
            plt.savefig(result_path+'train_reward_iter'+str(i)+'.png', bbox_inches='tight')
            plt.clf()

            plt.plot(test_in_train_case_reward_list)
            plt.title('test_in_train_case_reward')
            plt.savefig(result_path+'test_in_train_case_reward_iter'+str(i)+'.png', bbox_inches='tight')
            plt.clf()

            plt.plot(test_case_reward_list)
            plt.title('test case reward')
            plt.savefig(result_path+'test_case_reward_iter'+str(i)+'.png', bbox_inches='tight')
            plt.clf()

            if config_object['dataset'] == 'mario':
                plt.plot(test_task_reward_list)
                plt.title('test task reward')
                plt.savefig(result_path+'test_task_reward_iter'+str(i)+'.png', bbox_inches='tight')
                plt.clf()

            data = {
                'train reward': train_reward_list,
                'test_in_train_case_reward': test_in_train_case_reward_list,
                'test case reward': test_case_reward_list}

            if config_object['dataset'] == 'mario':
                data['test task reward'] = test_task_reward_list
            with open(result_path + 'record', 'wb') as f:
                pkl.dump(data, f)

    # ==================== plot ===========================

    plt.plot(train_reward_list)
    plt.title('train reward')
    plt.savefig(result_path + 'train_reward.png', bbox_inches='tight')
    plt.clf()

    plt.plot(test_in_train_case_reward_list)
    plt.title('test_in_train_case_reward')
    plt.savefig(result_path + 'test_in_train_case_reward.png', bbox_inches='tight')
    plt.clf()

    plt.plot(test_case_reward_list)
    plt.title('test case reward')
    plt.savefig(result_path + 'test_case_reward.png', bbox_inches='tight')
    plt.clf()

    if config_object['dataset'] == 'mario':
        plt.plot(test_task_reward_list)
        plt.title('test task reward')
        plt.savefig(result_path + 'test_task_reward.png', bbox_inches='tight')
        plt.clf()

    # ==================== save model ======================
    torch.save(agent, model_path + 'agent_final.pt')
