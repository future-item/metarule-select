import os
import numpy as np
import torch
import random

class PathManager:
    def __init__(self, path_object, dataset, exp_name):
        self.exp_root_path = path_object['exp_root_path']
        self.dataset_path = path_object['dataset_path']
        self.model_path = path_object['model_path']
        self.tmp_path = path_object['tmp_path']
        self.rule_path = path_object['rule_path']
        self.result_path = path_object['result_path']
        self.pl_tmp_path = path_object['pl_tmp_path']
        self.condition_gen_path = path_object['condition_gen_path']
        self.dataset = dataset
        self.exp_name = exp_name

    def get_spec_path(self, path_type):
        if path_type == 'model':
            path = os.path.join(self.exp_root_path, self.model_path, self.dataset, self.exp_name)
        elif path_type == 'pl':
            path = os.path.join(self.exp_root_path, self.pl_tmp_path, self.dataset, self.exp_name)
        elif path_type == 'result':
            path = os.path.join(self.exp_root_path, self.result_path, self.dataset, self.exp_name)
        elif path_type == 'gen':
            path = os.path.join(self.exp_root_path, self.condition_gen_path, self.dataset, self.exp_name)
        else:
            raise NameError('unsupported mode.')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_spec_file(self, path_type, file_name=None):
        if path_type == 'model':
            return os.path.join(self.exp_root_path, self.model_path, self.dataset, self.exp_name, file_name)
        elif path_type == 'pl':
            return os.path.join(self.exp_root_path, self.pl_tmp_path, self.dataset, self.exp_name, file_name)

    def get_gen_file(self, path_type, file_name):
        if path_type == 'dataset':
            return os.path.join(self.exp_root_path, self.dataset_path, file_name)


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)