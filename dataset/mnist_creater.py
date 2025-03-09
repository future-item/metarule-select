import os
from itertools import product, permutations
from PIL import Image
import numpy as np
import random as rd
from numpy import random
from tqdm import tqdm
from collections import defaultdict

def generate_alternative_result(original_result):
    while True:
        new_result = np.random.randint(0, 99)
        if new_result != original_result:
            return new_result

def generate_alternative_result_set(original_result_set):
    while True:
        new_result = np.random.randint(0, 99)
        if new_result not in original_result_set:
            return new_result

def generate_priority_neg(priority_sign, seq):
    seq_len = len(seq)
    if priority_sign == 'add':
        result = seq[0]
        if seq_len == 3:
            change_pos = 2
        else:
            change_pos = np.random.randint(2, seq_len-1)
        for j in range(1, seq_len):
            if j < change_pos:
                result *= seq[j]
            else:
                result += seq[j]
    else:
        result = seq[0]
        if seq_len == 3:
            change_pos = 2
        else:
            change_pos = np.random.randint(2, seq_len-1)
        for j in range(1, seq_len):
            if j < change_pos:
                result += seq[j]
            else:
                result *= seq[j]
    return result


def priority(priority_sign, len_list, num, pos=True):
    max_len = max(len_list)
    result_array = -np.ones((num, max_len + 1),dtype=np.int16)

    for i in range(num):
        seq_len = np.random.choice(len_list)
        sequence = np.random.randint(0, 9, seq_len)
        result_set = set()

        if priority_sign == 'add':
            for change_pos in range(1, seq_len+1):
                result = sequence[0]
                for j in range(1, seq_len):
                    if j < change_pos:
                        result += sequence[j]
                    else:
                        result *= sequence[j]
                result_set.add(result)
        elif priority_sign == 'multi':
            for change_pos in range(1, seq_len+1):
                result = sequence[0]
                for j in range(1, seq_len):
                    if j < change_pos:
                        result *= sequence[j]
                    else:
                        result += sequence[j]
                result_set.add(result)

        result_array[i, :seq_len] = sequence
        if pos:
            result_array[i, -1] = rd.choice(list(result_set))
        else:
            result_neg = generate_priority_neg(priority_sign, sequence)
            if result_neg not in result_set:
                result_array[i, -1] =  result_neg
            else:
                result_array[i, -1] = generate_alternative_result_set(result_set)
    return [result_array[i] for i in range(num)]

def cumulative(sign, len_list, num, pos=True, reverse=False):
    max_len = max(len_list)
    result_array = -np.ones((num, max_len + 1),dtype=np.int16)

    for i in range(num):
        seq_len = np.random.choice(len_list)
        sequence = np.random.randint(0, 9, seq_len)

        if sign == 'add':
            result = sequence[0]
            for j in range(1, seq_len):
                result += sequence[j]
        elif sign == 'multi':
            result = sequence[0]
            for j in range(1, seq_len):
                result *= sequence[j]

        if reverse:
            if seq_len != max_len:
                result_array[i, -max_len:-max_len + seq_len] = sequence
            else:
                result_array[i, -max_len:] = sequence
            if pos:
                result_array[i, 0] = result
            else:
                result_array[i, 0] = generate_alternative_result(result)
        else:
            result_array[i, :seq_len] = sequence
            if pos:
                result_array[i, -1] = result
            else:
                result_array[i, -1] = generate_alternative_result(result)
    return [result_array[i] for i in range(num)]

def increasing_sequence(len_list, num, pos=True):
    max_len = max(len_list)
    result_array = -np.ones((num, max_len + 1),dtype=np.int16)

    if pos:
        for i in range(num):
            seq_len = np.random.choice(len_list)
            sequence = np.sort(np.random.choice(range(10), seq_len, replace=False))
            result_array[i, :len(sequence)] = sequence
            result_array[i, -1] = 0
    else:
        for i in range(int(num*p_flag_true)):
            flag = True
            while flag:
                seq_len = np.random.choice(len_list)
                sequence = np.random.choice(range(10), seq_len, replace=False)
                sequence_alternative = np.sort(sequence)
                if not np.array_equal(sequence_alternative,sequence):
                    flag = False
            result_array[i, :len(sequence)] = sequence
            result_array[i, -1] = 0
        for i in range(int(num*p_flag_true),num):
            seq_len = np.random.choice(len_list)
            sequence = np.sort(np.random.choice(range(10), seq_len, replace=False))
            result_array[i, :len(sequence)] = sequence
            result_array[i, -1] = generate_alternative_result(0)
    return [result_array[i] for i in range(num)]

def decreasing_sequence(len_list, num, pos=True):
    max_len = max(len_list)
    result_array = -np.ones((num, max_len + 1), dtype=np.int16)

    if pos:
        for i in range(num):
            seq_len = np.random.choice(len_list)
            sequence = np.sort(np.random.choice(range(10), seq_len, replace=False))[::-1]
            if seq_len !=max_len:
                result_array[i, -max_len:-max_len+seq_len] = sequence
            else:
                result_array[i, -max_len:] = sequence
            result_array[i, 0] = 0
    else:
        for i in range(int(num*p_flag_true)):
            flag = True
            while flag:
                seq_len = np.random.choice(len_list)
                sequence = np.random.choice(range(10), seq_len, replace=False)
                sequence_alternative = np.sort(sequence)[::-1]
                if not np.array_equal(sequence_alternative,sequence):
                    flag = False
            if seq_len !=max_len:
                result_array[i, -max_len:-max_len+seq_len] = sequence
            else:
                result_array[i, -max_len:] = sequence
            result_array[i, 0] = 0
        for i in range(int(num*p_flag_true),num):
            seq_len = np.random.choice(len_list)
            sequence = np.sort(np.random.choice(range(10), seq_len, replace=False))[::-1]
            if seq_len !=max_len:
                result_array[i, -max_len:-max_len+seq_len] = sequence
            else:
                result_array[i, -max_len:] = sequence
            result_array[i, 0] = generate_alternative_result(0)
    return [result_array[i] for i in range(num)]

def sample_imgs(labels, reverse=False):
    if reverse:
        img_label = [labels[i][-max_pic_num:] for i in range(len(labels))]
    else:
        img_label = [labels[i][:max_pic_num] for i in range(len(labels))]
    imgs_return = []
    for i in range(len(img_label)):
        imgs = -np.ones((max_pic_num,28,28),dtype=np.int16)
        for j in range(len(img_label[i])):
            if img_label[i][j]!=-1:
                imgs[j] = rd.choice(source_image[img_label[i][j]])
        imgs_return.append(imgs)
    return imgs_return

if __name__ == '__main__':
    random.seed(888)
    mnist_dir = "./mnist_imgs"
    prev = "mnist_data"
    task_names = ['reverse_cumulative_product', 'reverse_cumulative_sum',
                  'decreasing_sequence', 'increasing_sequence', 'add_priority',
                  'multi_priority', 'cumulative_sum', 'cumulative_product']
    reverse_task = ['reverse_cumulative_sum', 'reverse_cumulative_product','decreasing_sequence']
    sign_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    max_pic_num = 5
    min_pic_num = 3
    num_pos_per_task = 2000
    num_neg_per_task = 2000
    p_flag_true = 0.5
    size = 28

    folder = f'/home/jiny/AbdGen_data/dataset/'
    file = f'mnist.npz'
    if not os.path.exists(folder):
        os.makedirs(folder)

    imgs_pos = {}
    lbls_pos = {}
    imgs_neg = {}
    lbls_neg = {}
    source_image = defaultdict(list)
    for sign in sign_list:
        source_image_path = mnist_dir+'/'+sign
        png_files = [f for f in os.listdir(source_image_path) if f.lower().endswith('.png')]
        source_image_name=[os.path.join(source_image_path, f) for f in png_files]
        for name in tqdm(source_image_name):
            with Image.open(name) as img:
                img_array = np.array(img, dtype=np.int16)
                source_image[int(sign)].append(img_array)
    #
    # np.savez('./mnist_source_imgs.npz', mnist_imgs=source_image)

    # source_image = np.load('./mnist_source_imgs.npz', encoding='latin1', allow_pickle=True)
    # source_image = source_image['mnist_imgs'].item()

    for task in tqdm(task_names):
        if task == 'add_priority':
            labels_fit_rule = priority('add',range(min_pic_num,max_pic_num+1),num_pos_per_task)
            labels_neg = priority('add', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)
        elif task == 'multi_priority':
            labels_fit_rule = priority('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task)
            labels_neg = priority('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)
        elif task == 'cumulative_sum':
            labels_fit_rule =cumulative('add', range(min_pic_num, max_pic_num + 1), num_pos_per_task)
            labels_neg = cumulative('add', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)
        elif task == 'cumulative_product':
            labels_fit_rule = cumulative('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task)
            labels_neg = cumulative('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)
        elif task == 'reverse_cumulative_sum':
            labels_fit_rule =cumulative('add', range(min_pic_num, max_pic_num + 1), num_pos_per_task,reverse=True)
            labels_neg = cumulative('add', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False,reverse=True)
        elif task == 'reverse_cumulative_product':
            labels_fit_rule = cumulative('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task,reverse=True)
            labels_neg = cumulative('multi', range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False,reverse=True)
        elif task == 'increasing_sequence':
            labels_fit_rule =increasing_sequence(range(min_pic_num, max_pic_num + 1), num_pos_per_task)
            labels_neg = increasing_sequence(range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)
        elif task == 'decreasing_sequence':
            labels_fit_rule = decreasing_sequence(range(min_pic_num, max_pic_num + 1), num_pos_per_task)
            labels_neg = decreasing_sequence(range(min_pic_num, max_pic_num + 1), num_pos_per_task, pos=False)

        if task in reverse_task:
            images_fit_rule = sample_imgs(labels_fit_rule,reverse=True)
            images_neg = sample_imgs(labels_neg,reverse=True)
        else:
            images_fit_rule = sample_imgs(labels_fit_rule)
            images_neg = sample_imgs(labels_neg)

        lbls_pos[task] = labels_fit_rule
        lbls_neg[task] = labels_neg
        imgs_pos[task] = images_fit_rule
        imgs_neg[task] = images_neg

    np.savez(os.path.join(folder, file),
             images_pos=imgs_pos,
             images_neg=imgs_neg,
             labels_pos=lbls_pos,
             labels_neg=lbls_neg)
