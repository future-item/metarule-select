import numpy as np
import asyncio

from utils.dataloader import MarioDataLoader, MnistDataLoader
from utils.metagol_pl_helper import task_writer_mario, bk_writer_mario, task_writer_mnist, bk_writer_mnist,run_pl, read_pl_out

class Mario_env:
    def __init__(self, config):
        self.loader = MarioDataLoader(config)
        self.max_pos = config['max_pos']
        self.max_neg = config['max_neg']
        self.num_metarule = config['num_metarule']
        self.max_pic_num = config['max_pic_num']
        self.error_reward = config['error_reward']
        self.B_len = config['B_len']
        self.num_action_sample = config['num_action_sample']
        self.noise = config['noise']
        self.noise_rate = config['noise_rate']
        self.time = config['pl_time_limit']

        self.log_file_path = config['log_file_path']
        self.pl_path = config['pl_path']

        # +1:frame type is not designed in B
        self.state_shape = (self.max_pos+self.max_neg, self.max_pic_num+self.B_len)
        self.action_shape = (self.num_metarule,)
        self.state = None
        self.ground_state = None
        self.task_name = None


    def get_state_shape(self):
        return self.state_shape

    def get_action_shape(self):
        return self.action_shape

    def add_noise(self, state):
        random_matrix = np.random.rand(*state.shape)
        random_values = np.random.randint(0, 10, state.shape)
        # make sure -1 is kept
        keep_mask = np.where(state == -1, True, random_matrix >= self.noise_rate)
        noisy_state = np.where(keep_mask, state, random_values)
        return noisy_state

    # sample a task
    def reset(self, task_name=None):
        # labels_pos:[pos_num_per_bag, label_size]
        # labels_neg:[neg_num_per_bag, label_size]
        if task_name is None:
            self.task_name, labels_pos, labels_neg = self.loader.get_train_data()
        else:
            self.task_name = task_name
            labels_pos, labels_neg = self.loader.get_train_data(self.task_name)
        task_writer_mario(self.task_name, labels_pos, labels_neg)
        # state include pos_case, neg_case and Metarule selected status
        # state:[pos_num_per_bag+neg_num_per_bag,size]
        labels_pos_padding = -np.ones((self.max_pos - labels_pos.shape[0],labels_pos.shape[1]))
        labels_neg_padding = -np.ones((self.max_neg - labels_neg.shape[0],labels_pos.shape[1]))
        self.state = np.concatenate((labels_pos, labels_pos_padding, labels_neg, labels_neg_padding), axis=0)
        self.state = self.state[:,:-1]
        if self.noise:
            self.state = np.concatenate((self.add_noise(self.state[:,:self.max_pic_num]),self.state[:,self.max_pic_num:]),axis=1)
        return self.state

    def reset_test(self, task_name):
        # labels_pos:[pos_num_per_bag, label_size]
        # labels_neg:[neg_num_per_bag, label_size]
        self.task_name = task_name
        labels_pos, labels_neg = self.loader.get_test_data(task_name)
        task_writer_mario(task_name, labels_pos, labels_neg)
        # state include pos_case, neg_case and Metarule selected status
        # state:[pos_num_per_bag+neg_num_per_bag,size]
        labels_pos_padding = -np.ones((self.max_pos - labels_pos.shape[0],labels_pos.shape[1]))
        labels_neg_padding = -np.ones((self.max_neg - labels_neg.shape[0],labels_pos.shape[1]))
        self.state = np.concatenate((labels_pos, labels_pos_padding, labels_neg, labels_neg_padding), axis=0)
        self.state = self.state[:,:-1]
        if self.noise:
            self.state = np.concatenate(
                (self.add_noise(self.state[:, :self.max_pic_num]), self.state[:, self.max_pic_num:]), axis=1)
        return self.state

    def step(self, action):
        reward_of_sample_actions = []
        for i in range(len(action)):
            bk_writer_mario(self.task_name, action[i])
            file_run = self.pl_path + 'task_' + self.task_name + '.pl'

            loop = asyncio.get_event_loop()
            task = loop.create_task(run_pl(file_run, self.time))
            loop.run_until_complete(task)

            pl_result, pl_out = task.result()
            with open(self.log_file_path, 'a') as f:
                if pl_result == -1:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nERROR\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == -2:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nTime out\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == 0:
                    # reward = -int(pow(2, np.sum(action[i])))
                    # reward = -int(np.sum(action[i]))
                    reward = int(pow(2, self.num_metarule-np.sum(action[i])))
                    result = self.task_name + ' sample No.' + str(i) + ':\n'
                    result = result + read_pl_out(pl_out) + '\n'
                    result = result + 'reward: '+str(reward)+'\n'
                    print(result,file=f)
                    print(result)
            reward_of_sample_actions.append(reward)
        return np.array(reward_of_sample_actions)

class Mnist_env:
    def __init__(self, config):
        self.loader = MnistDataLoader(config)
        self.reverse_task = config['reverse_task']
        self.max_pos = config['max_pos']
        self.max_neg = config['max_neg']
        self.num_metarule = config['num_metarule']
        self.max_pic_num = config['max_pic_num']
        self.error_reward = config['error_reward']
        self.B_len = config['B_len']
        self.num_action_sample = config['num_action_sample']
        self.noise = config['noise']
        self.noise_rate = config['noise_rate']
        self.time = config['pl_time_limit']

        self.log_file_path = config['log_file_path']
        self.pl_path = config['pl_path']

        # +1:frame type is not designed in B
        self.state_shape = (self.max_pos+self.max_neg, self.max_pic_num+self.B_len)
        self.action_shape = (self.num_metarule,)
        self.state = None
        self.ground_state = None
        self.task_name = None


    def get_state_shape(self):
        return self.state_shape

    def add_noise(self, state):
        random_matrix = np.random.rand(*state.shape)
        random_values = np.random.randint(0, 10, state.shape)
        # make sure -1 is kept
        keep_mask = np.where(state == -1, True, random_matrix >= self.noise_rate)
        noisy_state = np.where(keep_mask, state, random_values)
        return noisy_state

    # sample a task
    def reset(self, task_name=None):
        # labels_pos:[pos_num_per_bag, label_size]
        # labels_neg:[neg_num_per_bag, label_size]
        if task_name is None:
            self.task_name, labels_pos, labels_neg = self.loader.get_train_data()
        else:
            self.task_name = task_name
            labels_pos, labels_neg = self.loader.get_train_data(self.task_name)
        task_writer_mnist(self.task_name, labels_pos, labels_neg)
        # state include pos_case, neg_case and Metarule selected status
        # state:[pos_num_per_bag+neg_num_per_bag,size]
        labels_pos_padding = -np.ones((self.max_pos - labels_pos.shape[0],labels_pos.shape[1]))
        labels_neg_padding = -np.ones((self.max_neg - labels_neg.shape[0],labels_pos.shape[1]))
        self.state = np.concatenate((labels_pos, labels_pos_padding, labels_neg, labels_neg_padding), axis=0)
        if self.noise and task_name not in self.reverse_task:
            self.state = np.concatenate((self.add_noise(self.state[:, :self.max_pic_num]), self.state[:, self.max_pic_num:]), axis=1)
        elif self.noise and task_name in self.reverse_task:
            self.state = np.concatenate((self.state[:, :self.B_len], self.add_noise(self.state[:, self.B_len:])), axis=1)
        return self.state

    def reset_test(self, task_name):
        # labels_pos:[pos_num_per_bag, label_size]
        # labels_neg:[neg_num_per_bag, label_size]
        self.task_name = task_name
        labels_pos, labels_neg = self.loader.get_test_data(task_name)
        task_writer_mnist(task_name, labels_pos, labels_neg)
        # state include pos_case, neg_case and Metarule selected status
        # state:[pos_num_per_bag+neg_num_per_bag,size]
        labels_pos_padding = -np.ones((self.max_pos - labels_pos.shape[0],labels_pos.shape[1]))
        labels_neg_padding = -np.ones((self.max_neg - labels_neg.shape[0],labels_pos.shape[1]))
        self.state = np.concatenate((labels_pos, labels_pos_padding, labels_neg, labels_neg_padding), axis=0)
        if self.noise and task_name not in self.reverse_task:
            self.state = np.concatenate((self.add_noise(self.state[:, :self.max_pic_num]), self.state[:, self.max_pic_num:]), axis=1)
        elif self.noise and task_name in self.reverse_task:
            self.state = np.concatenate((self.state[:, :self.B_len], self.add_noise(self.state[:, self.B_len:])), axis=1)
        return self.state

    def step(self, action):
        reward_of_sample_actions = []
        for i in range(len(action)):
            bk_writer_mnist(self.task_name, action[i])
            file_run = self.pl_path + 'task_' + self.task_name + '.pl'

            loop = asyncio.get_event_loop()
            task = loop.create_task(run_pl(file_run, self.time))
            loop.run_until_complete(task)

            pl_result, pl_out = task.result()
            with open(self.log_file_path, 'a') as f:
                if pl_result == -1:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nERROR\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == -2:
                    reward = self.error_reward
                    result = self.task_name + ' sample No.' + str(i) + ':\nTime out\n'
                    result = result + 'reward: ' + str(reward)+'\n'
                    print(result,file=f)
                    print(result)
                elif pl_result == 0:
                    # reward = -int(pow(2, np.sum(action[i])))
                    # reward = -int(np.sum(action[i]))
                    # pl_str, pl_len= read_pl_out(pl_out, need_len=True)
                    # reward = int(pow(2, self.num_metarule-np.sum(action[i]))+16-4*pl_len)
                    pl_str = read_pl_out(pl_out)
                    reward = int(pow(2, self.num_metarule - np.sum(action[i])))
                    result = self.task_name + ' sample No.' + str(i) + ':\n'
                    result = result + pl_str + '\n'
                    result = result + 'reward: '+str(reward)+'\n'
                    print(result,file=f)
                    print(result)
            reward_of_sample_actions.append(reward)
        return np.array(reward_of_sample_actions)
