import numpy as np
import random as rd

class MarioDataLoader:
    def __init__(self, config, path_manager=None, mode='PPO'):
        self.len_guarantee_task = config['len_guarantee_task']
        self.all_task = config['all_task']
        self.max_pic_num = config['max_pic_num']
        self.grd_map = config['grd_map']

        if mode == 'AbdGen':
            self.task = config['task']
            # dataset_path = path_manager.get_gen_file('dataset', 'mario.npz')
            dataset_path = path_manager.get_gen_file('dataset', 'mario_'+self.task+'.npz')
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.img_size = config['input_size']
            self.imgs_pos = self.data_zip['images_pos'].item()[self.task]
            self.imgs_neg = self.data_zip['images_neg'].item()[self.task]
            self.lbls_pos = self.data_zip['labels_pos'].item()[self.task]
            self.lbls_neg = self.data_zip['labels_neg'].item()[self.task]

            if self.task in self.len_guarantee_task:
                len_guarantee_index_pos = [index for index in range(len(self.lbls_pos)) if len(self.lbls_pos[index][0]) > 2]
                len_guarantee_index_neg = [index for index in range(len(self.lbls_neg)) if len(self.lbls_neg[index][0]) > 2]
                self.imgs_pos = [self.imgs_pos[len_guarantee_index_pos[i]] for i in range(len(len_guarantee_index_pos))]
                self.imgs_neg = [self.imgs_neg[len_guarantee_index_neg[i]] for i in range(len(len_guarantee_index_neg))]
                self.lbls_pos = [self.lbls_pos[len_guarantee_index_pos[i]] for i in range(len(len_guarantee_index_pos))]
                self.lbls_neg = [self.lbls_neg[len_guarantee_index_neg[i]] for i in range(len(len_guarantee_index_neg))]

            self.img_mean = np.mean(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                               np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)
            self.img_std = np.std(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                                        np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)

            self.num_bag_per_batch = config['num_bag_per_batch']
            self.num_pos_per_bag = config['num_pos_per_bag']
            self.num_neg_per_bag = config['num_neg_per_bag']

            combined_pos = list(zip(self.imgs_pos, self.lbls_pos))
            combined_neg = list(zip(self.imgs_neg, self.lbls_neg))
            rd.shuffle(combined_pos)
            rd.shuffle(combined_neg)
            self.imgs_pos, self.lbls_pos = zip(*combined_pos)
            self.imgs_neg, self.lbls_neg = zip(*combined_neg)

        elif mode == 'AbdGen_pretrain':
            self.task = config['task']
            dataset_path = path_manager.get_gen_file('dataset', 'mario_'+self.task+'_pretrain.npz')
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.img_size = config['input_size']
            self.imgs = self.data_zip['images_pre'].item()[self.task]
            self.lbls = self.data_zip['labels_pre'].item()[self.task]

            self.img_mean = np.mean(np.concatenate(self.imgs, axis=0),axis=0)
            self.img_std = np.std(np.concatenate(self.imgs, axis=0), axis=0)

            for i in range(len(self.lbls)):
                self.lbls[i] = self.grd_map[self.lbls[i][0]]
                self.imgs[i] = (self.imgs[i].astype(np.float32) - self.img_mean) / self.img_std

            self.num_img_per_batch = config['num_img_per_batch_pre_train']

            combined = list(zip(self.imgs, self.lbls))
            rd.shuffle(combined)
            self.imgs, self.lbls = zip(*combined)
            print('pic_num: '+str(len(self.lbls)))

        elif mode == 'PPO':
            dataset_path = config['data_path']
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.train_task = config['train_task']
            self.test_task = config['test_task']
            self.train_task_num = len(self.train_task)
            self.test_task_num = len(self.test_task)
            self.lbls_pos = self.data_zip['labels_pos'].item()
            self.lbls_neg = self.data_zip['labels_neg'].item()

            self.num_neg_per_task = len(self.lbls_neg[self.train_task[0]])
            self.pos_train_num = {}
            self.pos_test_num = {}
            for key in self.lbls_pos:
                self.pos_test_num[key] = int(len(self.lbls_pos[key]) * config['test_p'])
                self.pos_train_num[key] = len(self.lbls_pos[key]) - self.pos_test_num[key]
            self.neg_test_num = int(self.num_neg_per_task * config['test_p'])
            self.neg_train_num = self.num_neg_per_task - self.neg_test_num

            self.test_pos_num = {}
            for task_name in self.test_task:
                self.test_pos_num[task_name] = len(self.lbls_pos[task_name])

            self.num_pos = config['num_pos']
            self.num_neg = config['num_neg']

            for key in self.lbls_pos:
                rd.shuffle(self.lbls_pos[key])
                rd.shuffle(self.lbls_neg[key])


    def grd_label_to_array_label(self, case):
        array_label = np.zeros((self.max_pic_num + 4), dtype=np.int16)
        for i in range(self.max_pic_num):
            if i < len(case[0]):
                array_label[i] = self.grd_map[case[0][i]]
            else:
                array_label[i] = -1
        array_label[self.max_pic_num] = self.grd_map[case[1][0]]
        for i in range(1, 4):
            array_label[self.max_pic_num + i] = case[1][i]
        return array_label

    def img_to_max_len(self, img_case):
        padding_imgs = -np.ones((self.max_pic_num, self.img_size[0], self.img_size[1], self.img_size[2]), dtype=np.float32)
        for i in range(self.max_pic_num):
            if i < len(img_case):
                padding_imgs[i] = (img_case[i].astype(np.float32)-self.img_mean)/self.img_std
        return padding_imgs

    def get_train_data(self, task_name=None):
        # sample task
        if task_name is None:
            task_name = rd.choices(self.train_task, k=1)[0]
            labels_pos, labels_neg = self.get_data_train_task(task_name, 'train')
            return  task_name, labels_pos, labels_neg
        else:
            labels_pos, labels_neg = self.get_data_train_task(task_name, 'train')
            return labels_pos, labels_neg

    def get_test_data(self, task_name):
        if task_name in self.test_task:
            labels_pos, labels_neg = self.get_data_test_task(task_name)
        else:
            labels_pos, labels_neg = self.get_data_train_task(task_name, 'test')
        return  labels_pos, labels_neg

    def get_data_train_task(self, task_name, mode):
        # sample positive and negative cases for this task
        pos_train_num = self.pos_train_num[task_name]
        n_pos = np.random.choice(self.num_pos, 1)
        n_neg = np.random.choice(self.num_neg, 1)
        if mode == 'train':
            pos_case_index = np.random.choice(range(pos_train_num), n_pos, replace=False)
            neg_case_index = np.random.choice(range(self.neg_train_num), n_neg, replace=False)
        else:
            pos_test_num = self.pos_test_num[task_name]
            pos_case_index = np.random.choice(range(pos_train_num, pos_train_num+pos_test_num), n_pos, replace=False)
            neg_case_index = np.random.choice(range(self.neg_train_num, self.neg_train_num+self.neg_test_num), n_neg, replace=False)

        # priority and jump task should guarantee at least one longest case is chosen
        if task_name in self.len_guarantee_task:
            flag = True
            while flag:
                for index in pos_case_index:
                    if len(self.lbls_pos[task_name][index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag and mode == 'train':
                    pos_case_index = np.random.choice(range(pos_train_num), n_pos, replace=False)
                elif flag and mode =='test':
                    pos_case_index = np.random.choice(range(pos_train_num, pos_train_num + pos_test_num), n_pos, replace=False)

        labels_pos = [self.grd_label_to_array_label(self.lbls_pos[task_name][i]) for i in pos_case_index]
        labels_neg = [self.grd_label_to_array_label(self.lbls_neg[task_name][i]) for i in neg_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)
        return labels_pos, labels_neg

    def get_data_test_task(self, task_name):
        # sample positive and negative cases for this test task
        pos_num = self.test_pos_num[task_name]
        n_pos = np.random.choice(self.num_pos, 1)
        n_neg = np.random.choice(self.num_neg, 1)
        pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)
        neg_case_index = np.random.choice(range(self.num_neg_per_task), n_neg, replace=False)

        # priority and jump task should guarantee at least one longest case is chosen
        if task_name in self.len_guarantee_task:
            flag = True
            while flag:
                for index in pos_case_index:
                    if len(self.lbls_pos[task_name][index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)

        labels_pos = [self.grd_label_to_array_label(self.lbls_pos[task_name][i]) for i in pos_case_index]
        labels_neg = [self.grd_label_to_array_label(self.lbls_neg[task_name][i]) for i in neg_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)
        return labels_pos, labels_neg

    def get_data_AbdGen(self):
        # sample positive and negative cases for this test task
        pos_num = len(self.lbls_pos)
        neg_num = len(self.lbls_neg)
        n_pos = self.num_pos_per_bag
        n_neg = self.num_neg_per_bag
        pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)
        neg_case_index = np.random.choice(range(neg_num), n_neg, replace=False)

        # priority task should guarantee at least one longest case is chosen
        if self.task in self.len_guarantee_task:
            flag = True
            while flag:
                for index in pos_case_index:
                    if len(self.lbls_pos[index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)
            flag = True
            while flag:
                for index in neg_case_index:
                    if len(self.lbls_neg[index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    neg_case_index = np.random.choice(range(neg_num), n_neg, replace=False)

        labels_pos = [self.grd_label_to_array_label(self.lbls_pos[i]) for i in pos_case_index]
        labels_neg = [self.grd_label_to_array_label(self.lbls_neg[i]) for i in neg_case_index]

        imgs_pos = [self.img_to_max_len(self.imgs_pos[i]) for i in pos_case_index]
        imgs_neg = [self.img_to_max_len(self.imgs_neg[i]) for i in neg_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)

        imgs_pos = np.array(imgs_pos)
        imgs_neg = np.array(imgs_neg)

        return imgs_pos, labels_pos, imgs_neg, labels_neg

    def get_data_AbdGen_pretrain(self):
        img_num = len(self.lbls)
        n_img = self.num_img_per_batch
        img_index = np.random.choice(range(img_num), n_img, replace=True)

        labels = [self.lbls[i] for i in img_index]
        imgs = [self.imgs[i] for i in img_index]

        labels = np.array(labels)
        imgs = np.array(imgs)

        return imgs, labels


class MnistDataLoader:
    def __init__(self, config, path_manager=None, mode='PPO'):
        self.all_task = config['all_task']
        self.max_pic_num = config['max_pic_num']

        if mode == 'AbdGen':
            self.task = config['task']
            dataset_path = path_manager.get_gen_file('dataset', 'mnist.npz')
            # dataset_path = path_manager.get_gen_file('dataset', 'mnist_'+self.task+'.npz')
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.img_size = config['input_size']
            self.imgs_pos = self.data_zip['images_pos'].item()[self.task]
            self.imgs_neg = self.data_zip['images_neg'].item()[self.task]
            self.lbls_pos = self.data_zip['labels_pos'].item()[self.task]
            self.lbls_neg = self.data_zip['labels_neg'].item()[self.task]

            self.img_mean = np.mean(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                               np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)
            self.img_std = np.std(np.concatenate([np.concatenate(self.imgs_pos, axis=0),
                                                        np.concatenate(self.imgs_neg, axis=0)], axis=0), axis=0)

            self.num_bag_per_batch = config['num_bag_per_batch']
            self.num_pos_per_bag = config['num_pos_per_bag']
            self.num_neg_per_bag = config['num_neg_per_bag']

            combined_pos = list(zip(self.imgs_pos, self.lbls_pos))
            combined_neg = list(zip(self.imgs_neg, self.lbls_neg))
            rd.shuffle(combined_pos)
            rd.shuffle(combined_neg)
            self.imgs_pos, self.lbls_pos = zip(*combined_pos)
            self.imgs_neg, self.lbls_neg = zip(*combined_neg)

        elif mode == 'AbdGen_pretrain':
            self.task = config['task']
            dataset_path = path_manager.get_gen_file('dataset', 'mario_'+self.task+'_pretrain.npz')
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.img_size = config['input_size']
            self.imgs = self.data_zip['images_pre'].item()[self.task]
            self.lbls = self.data_zip['labels_pre'].item()[self.task]

            self.img_mean = np.mean(np.concatenate(self.imgs, axis=0),axis=0)
            self.img_std = np.std(np.concatenate(self.imgs, axis=0), axis=0)

            for i in range(len(self.lbls)):
                self.lbls[i] = self.grd_map[self.lbls[i][0]]
                self.imgs[i] = (self.imgs[i].astype(np.float32) - self.img_mean) / self.img_std

            self.num_img_per_batch = config['num_img_per_batch_pre_train']

            combined = list(zip(self.imgs, self.lbls))
            rd.shuffle(combined)
            self.imgs, self.lbls = zip(*combined)
            print('pic_num: '+str(len(self.lbls)))

        elif mode == 'PPO':
            dataset_path = config['data_path']
            self.data_zip = np.load(dataset_path, encoding='latin1', allow_pickle=True)
            self.task_num = len(self.all_task)
            self.lbls_pos = self.data_zip['labels_pos'].item()
            self.lbls_neg = self.data_zip['labels_neg'].item()

            self.num_neg_per_task = len(self.lbls_neg[self.all_task[0]])
            self.num_pos_per_task = len(self.lbls_pos[self.all_task[0]])
            self.pos_test_num = int(self.num_pos_per_task * config['test_p'])
            self.pos_train_num = self.num_pos_per_task - self.pos_test_num
            self.neg_test_num = int(self.num_neg_per_task * config['test_p'])
            self.neg_train_num = self.num_neg_per_task - self.neg_test_num

            self.num_pos = config['num_pos']
            self.num_neg = config['num_neg']

    def get_train_data(self, task_name=None):
        if task_name is None:
            task_name = rd.choices(self.all_task, k=1)[0]
            labels_pos, labels_neg = self.get_data_train_task(task_name, 'train')
            return  task_name, labels_pos, labels_neg
        else:
            labels_pos, labels_neg = self.get_data_train_task(task_name, 'train')
            return labels_pos, labels_neg

    def get_test_data(self, task_name):
        labels_pos, labels_neg = self.get_data_train_task(task_name, 'test')
        return  labels_pos, labels_neg

    def get_data_train_task(self, task_name, mode):
        pos_train_num = self.pos_train_num
        n_pos = np.random.choice(self.num_pos, 1)
        n_neg = np.random.choice(self.num_neg, 1)
        if mode == 'train':
            pos_case_index = np.random.choice(range(pos_train_num), n_pos, replace=False)
            neg_case_index = np.random.choice(range(self.neg_train_num), n_neg, replace=False)
        else:
            pos_test_num = self.pos_test_num
            pos_case_index = np.random.choice(range(pos_train_num, pos_train_num+pos_test_num), n_pos, replace=False)
            neg_case_index = np.random.choice(range(self.neg_train_num, self.neg_train_num+self.neg_test_num), n_neg, replace=False)

        labels_pos = [self.lbls_pos[task_name][i] for i in pos_case_index]
        labels_neg = [self.lbls_neg[task_name][i] for i in neg_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)
        return labels_pos, labels_neg

    def get_data_AbdGen(self):
        # sample positive and negative cases for this test task
        pos_num = len(self.lbls_pos)
        neg_num = len(self.lbls_neg)
        n_pos = self.num_pos_per_bag
        n_neg = self.num_neg_per_bag
        pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)
        neg_case_index = np.random.choice(range(neg_num), n_neg, replace=False)

        # priority task should guarantee at least one longest case is chosen
        if self.task in self.len_guarantee_task:
            flag = True
            while flag:
                for index in pos_case_index:
                    if len(self.lbls_pos[index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    pos_case_index = np.random.choice(range(pos_num), n_pos, replace=False)
            flag = True
            while flag:
                for index in neg_case_index:
                    if len(self.lbls_neg[index][0]) == self.max_pic_num:
                        flag =False
                        break
                if flag:
                    neg_case_index = np.random.choice(range(neg_num), n_neg, replace=False)

        labels_pos = [self.grd_label_to_array_label(self.lbls_pos[i]) for i in pos_case_index]
        labels_neg = [self.grd_label_to_array_label(self.lbls_neg[i]) for i in neg_case_index]

        imgs_pos = [self.img_to_max_len(self.imgs_pos[i]) for i in pos_case_index]
        imgs_neg = [self.img_to_max_len(self.imgs_neg[i]) for i in neg_case_index]

        labels_pos = np.array(labels_pos)
        labels_neg = np.array(labels_neg)

        imgs_pos = np.array(imgs_pos)
        imgs_neg = np.array(imgs_neg)

        return imgs_pos, labels_pos, imgs_neg, labels_neg

    def get_data_AbdGen_pretrain(self):
        img_num = len(self.lbls)
        n_img = self.num_img_per_batch
        img_index = np.random.choice(range(img_num), n_img, replace=True)

        labels = [self.lbls[i] for i in img_index]
        imgs = [self.imgs[i] for i in img_index]

        labels = np.array(labels)
        imgs = np.array(imgs)

        return imgs, labels