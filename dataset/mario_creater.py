import os
from itertools import product, permutations

import numpy as np
from PIL import Image
from matplotlib import use
import random as rd
from numpy import random
from tqdm import tqdm

# Disable canvas visualization
use('Agg')

# =================== prepare icons ===========================
agents = ['mario']
targets = ['coin', 'bomb']
backgrounds = ['flowers1', 'sea', 'chessboard_pink']
frames = ['brick', 'brick2', 'brick3', 'green_panel', 'white_panel', 'glass', 'concrete']

ICONS = {
    'glass': f'./mario_icons/glass.png',
    'flowers1': f'./mario_icons/flowers1.png',
    'flowers2': f'./mario_icons/flowers2.png',
    'brick': f'./mario_icons/brick.png',
    'brick2': f'./mario_icons/brick2.png',
    'brick3': f'./mario_icons/brick3.png',
    'concrete': f'./mario_icons/concrete.png',
    'wood': f'./mario_icons/wood.png',
    'white_panel': f'./mario_icons/white_panel.png',
    'green_panel': f'./mario_icons/green_panel.png',

    'lava': f'./mario_icons/lava.png',
    'sea': f'./mario_icons/sea.png',
    'sand': f'./mario_icons/sand.png',
    'grass': f'./mario_icons/grass.png',
    'chessboard': f'./mario_icons/chessboard.png',
    'chessboard_blue': f'./mario_icons/chessboard_blue.png',
    'chessboard_pink': f'./mario_icons/chessboard_pink.png',
    'brindle': f'./mario_icons/brindle.png',

    'mario': f'./mario_icons/mario.png',
    'luigi': f'./mario_icons/luigi.png',
    'peach': f'./mario_icons/peach.png',
    'bomb': f'./mario_icons/bomb.png',
    'goomba': f'./mario_icons/goomba.png',

    'green_mushroom': f'./mario_icons/green_mushroom.png',
    'star': f'./mario_icons/star.png',
    'red_mushroom': f'./mario_icons/red_mushroom.png',
    'coin': f'./mario_icons/coin.png',
    'cloud': f'./mario_icons/cloud.png'
}

def PNG_ResizeKeepTransparency(img, new_width=0, new_height=0, resample="LANCZOS", RefFile=''):
    # needs PIL
    # Inputs:
    #   - SourceFile  = initial PNG file (including the path)
    #   - ResizedFile = resized PNG file (including the path)
    #   - new_width   = resized width in pixels; if you need % plz include it here: [your%] *initial width
    #   - new_height  = resized hight in pixels ; default = 0 = it will be calculated using new_width
    #   - resample = "NEAREST", "BILINEAR", "BICUBIC" and "ANTIALIAS"; default = "ANTIALIAS"
    #   - RefFile  = reference file to get the size for resize; default = ''

    img = img.convert("RGBA")  # convert to RGBA channels
    width, height = img.size  # get initial size

    # if there is a reference file to get the new size
    if RefFile != '':
        imgRef = Image.open(RefFile)
        new_width, new_height = imgRef.size
    else:
        # if we use only the new_width to resize in proportion the new_height
        # if you want % of resize please use it into new_width (?% * initial width)
        if new_height == 0:
            new_height = new_width * width / height

    # split image by channels (bands) and resize by channels
    img.load()
    bands = img.split()
    # resample mode
    if resample == "NEAREST":
        resample = Image.NEAREST
    else:
        if resample == "BILINEAR":
            resample = Image.BILINEAR
        else:
            if resample == "BICUBIC":
                resample = Image.BICUBIC
            else:
                if resample == "ANTIALIAS":
                    resample = Image.ANTIALIAS
                else:
                    if resample == "LANCZOS":
                        resample = Image.LANCZOS
    bands = [b.resize((new_width, new_height), resample) for b in bands]
    # merge the channels after individual resize
    img = Image.merge('RGBA', bands)
    return img


def draw_mario_world(X, Y, agent_x, agent_y, target_x, target_y, agent_icon='goomba', target_icon='green_mushroom',
                     background_tile='lava', frame_tile='glass'):
    """
    This method creates the specified Mario's world.
    """
    # Initialize canvas
    W, H = 20, 20
    image = Image.new("RGBA", ((X + 2) * W, (Y + 2) * H), (255, 255, 255))
    # Define y offset for PIL
    agent_y = - (agent_y - (Y - 1))
    target_y = - (target_y - (Y - 1))
    # Set off-set due to frame_dict
    agent_x, agent_y = agent_x + 1, agent_y + 1
    target_x, target_y = target_x + 1, target_y + 1
    # Scale position to tile dimension
    agent_x, agent_y = agent_x * W, agent_y * H
    target_x, target_y = target_x * W, target_y * H
    # Load mario_icons and tiles
    agent_icon = Image.open(ICONS[agent_icon])
    target_icon = Image.open(ICONS[target_icon])
    background_tile = Image.open(ICONS[background_tile])
    frame_tile = Image.open(ICONS[frame_tile])
    # Resize mario_icons and tiles to fit the image

    background_tile = background_tile.resize((W, H), Image.LANCZOS)
    frame_tile = frame_tile.resize((W, H), Image.LANCZOS)
    agent_icon = PNG_ResizeKeepTransparency(agent_icon, new_width=int(W / 2), new_height=int(H / 2), resample="LANCZOS", RefFile='')
    target_icon = PNG_ResizeKeepTransparency(target_icon, new_width=int(W / 2) + 2, new_height=int(H / 2) + 2, resample="LANCZOS", RefFile='')
    # Define frame_dict tiles left corners
    frame_tiles_pos = []
    for i in range(Y + 2):
        frame_tiles_pos.append((0, i * H))
        frame_tiles_pos.append(((X+1) * W, i * H))
        frame_tiles_pos.append((i * W, 0))
        frame_tiles_pos.append((i * W, (X+1) * H))
    # Define background_dict tiles left corners
    bkg_tiles_pos = []
    for i in range(1, Y + 1):
        for j in range(1, X + 1):
            bkg_tiles_pos.append((j * W, i * H))
    # Draw frame_dict
    for box in frame_tiles_pos:
        image.paste(frame_tile, box=box)
    # Draw background_dict
    for box in bkg_tiles_pos:
        image.paste(background_tile, box=box)
    # Draw target_dict
    target_box = (target_x + 4, target_y + 4)
    image.paste(target_icon, box=target_box, mask=target_icon)
    # Draw agent_dict
    agent_box = (agent_x + 5, agent_y + 5)
    image.paste(agent_icon, box=agent_box, mask=agent_icon)

    return np.array(image)[:, :, :3]

# pos1:(x1,y1)
# pos2:(x2,y2)
def define_move(pos1, pos2):
    if pos1[0] == pos2[0]:
        if pos2[1] == pos1[1] + 1:
            return 'up'
        elif pos2[1] == pos1[1] - 1:
            return 'down'
    elif pos1[1] == pos2[1]:
        if pos2[0] == pos1[0] + 1:
            return 'right'
        elif pos2[0] == pos1[0] - 1:
            return 'left'
    elif abs(pos2[0]-pos1[0]) == 1 and abs(pos2[1]-pos1[1]) == 1:
        return 'jump'
    return None

def add_target_pos(label):
    label[1] = [label[0][-1]] + label[1]
    return label

def move_target_pos(label):
    label[1] = [label[0][-1]] + label[1]
    label[0] = label[0][0:-1]
    return label

def priority(w, h, B_size, direction):
    A_list = []
    for x_start, y_start in product(range(w), range(h)):
        for x_finish, y_finish in product(range(w), range(h)):
            if direction in ['right', 'up'] and \
                    (x_finish - x_start < 0 or y_finish - y_start < 0):
                continue
            if direction in ['left', 'down'] and \
                    (x_finish - x_start > 0 or y_finish - y_start > 0):
                continue
            if x_finish - x_start == 0 and y_finish - y_start == 0:
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            if direction in ['right','left']:
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
            if direction in ['down', 'up']:
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
            A_list.append(pos)
    B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def just(w, h, B_size, direction):
    A_list = []
    for x_start, y_start in product(range(w), range(h)):
        for x_finish, y_finish in product(range(w), range(h)):
            if direction == 'right' and (y_finish != y_start or x_finish <= x_start):
                continue
            if direction == 'left' and (y_finish != y_start or x_finish >= x_start):
                continue
            if direction == 'up' and (x_finish != x_start or y_finish <= y_start):
                continue
            if direction == 'down' and (x_finish != x_start or y_finish >= y_start):
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            if direction in ['right','left']:
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
            if direction in ['down', 'up']:
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
            A_list.append(pos)
    B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def onestep(w, h, B_size, direction):
    A_list = []
    for x_start, y_start in product(range(w), range(h)):
        for x_finish, y_finish in product(range(w), range(h)):
            if direction == 'right' and (y_finish != y_start or x_finish <= x_start):
                continue
            if direction == 'left' and (y_finish != y_start or x_finish >= x_start):
                continue
            if direction == 'up' and (x_finish != x_start or y_finish <= y_start):
                continue
            if direction == 'down' and (x_finish != x_start or y_finish >= y_start):
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            if direction in ['right','left']:
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
            if direction in ['down', 'up']:
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
            if len(pos)==2:
                A_list.append(pos)
    B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def sea(w, h, B_size):
    A_list = []
    for x_start, y_start in product(range(w), range(h)):
        for x_finish, y_finish in product(range(w), range(h)):
            if x_finish != x_start or y_finish >= y_start:
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            while y_finish != y_now:
                y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                pos.append((int(x_now), int(y_now)))
            A_list.append(pos)

    B_list = list(map(list, product(range(B_size[1]), [1], range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def far(w, h, B_size):
    A_list = []
    for x_1, y_1 in product(range(w), range(h)):
        for x_2, y_2 in product(range(w), range(h)):
            for x_target, y_target in product(range(w), range(h)):
                if abs(x_target - x_1) + abs(y_target - y_1) < \
                        abs(x_target - x_2) + abs(y_target - y_2):
                    pos = [(x_1,y_1),(x_2,y_2),(x_target,y_target)]
                    A_list.append(pos)
    B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(move_target_pos, labels))
    return labels

def bomb_far(w, h, B_size):
    A_list = []
    for x_1, y_1 in product(range(w), range(h)):
        for x_2, y_2 in product(range(w), range(h)):
            for x_target, y_target in product(range(w), range(h)):
                if abs(x_target - x_1) + abs(y_target - y_1) < \
                        abs(x_target - x_2) + abs(y_target - y_2):
                    pos = [(x_1,y_1),(x_2,y_2),(x_target,y_target)]
                    A_list.append(pos)
    B_list = list(map(list, product([1], range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(move_target_pos, labels))
    return labels

def flower(w, h, B_size):
    A_list = []
    direction = 'right'
    for x_start, y_start in product(range(w), range(h)):
        for x_finish, y_finish in product(range(w), range(h)):
            if direction in ['right', 'up'] and \
                    (x_finish - x_start < 0 or y_finish - y_start < 0):
                continue
            if direction in ['left', 'down'] and \
                    (x_finish - x_start > 0 or y_finish - y_start > 0):
                continue
            if x_finish - x_start == 0 and y_finish - y_start == 0:
                continue
            pos = list()
            x_now = x_start
            y_now = y_start
            pos.append((x_now, y_now))
            if direction in ['right','left']:
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
            if direction in ['down', 'up']:
                while y_finish != y_now:
                    y_now = y_now + (y_finish - y_now)/abs(y_finish - y_now)
                    pos.append((int(x_now), int(y_now)))
                while x_finish != x_now:
                    x_now = x_now + (x_finish - x_now)/abs(x_finish - x_now)
                    pos.append((int(x_now), int(y_now)))
            A_list.append(pos)
    B_list = list(map(list, product(range(B_size[1]), [0], range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def chess_jump(w, h, pic_num, B_size):
    pos_list = []
    for x in range(w):
        for y in range(h):
            pos_list.append((x, y))

    A_list = []
    for n in range(2, pic_num + 1):
        A_list = A_list + list(product(pos_list, repeat=n))

    A_list_jump = []
    for j in range(len(A_list)):
        flag = True
        A = list(A_list[j])
        for k in range(len(A) - 1):
            movement = define_move(A[k], A[k + 1])
            if movement != 'jump':
                flag = False
                break
        if not flag:
            continue
        A_list_jump.append(A)

    B_list = list(map(list, product(range(B_size[1]), [2], range(B_size[3]))))
    labels = list(map(list, product(A_list_jump, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def generate_all_labels(w, h, pic_num, B_size):
    pos_list = []
    for x in range(w):
        for y in range(h):
            pos_list.append((x, y))

    A_list = []
    for n in range(2, pic_num + 1):
        A_list = A_list + list(map(list, permutations(pos_list, n)))

    B_list = list(map(list, product(pos_list, range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list, B_list)))
    return labels

def generate_all_path_labels(w, h, pic_num, B_size, background=None, movement_list=['right','up']):
    pos_list = []
    for x in range(w):
        for y in range(h):
            pos_list.append((x, y))

    A_list = []
    for n in range(2, pic_num + 1):
        A_list = A_list + list(map(list, permutations(pos_list, n)))

    A_list_path = []
    for j in range(len(A_list)):
        flag = True
        A = list(A_list[j])
        # No repetition point
        A_set = set(A)
        if len(A)!=len(A_set):
            continue
        for k in range(len(A) - 1):
            movement = define_move(A[k], A[k + 1])
            if movement not in movement_list:
                flag = False
                break
        if not flag:
            continue
        A_list_path.append(A)
    if background is None:
        B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    elif background=='sea':
        B_list = list(map(list, product(range(B_size[1]), [1], range(B_size[3]))))
    elif background=='flower':
        B_list = list(map(list, product(range(B_size[1]), [0], range(B_size[3]))))

    labels = list(map(list, product(A_list_path, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels


def generate_all_jump_labels(w, h, pic_num, B_size):
    pos_list = []
    for x in range(w):
        for y in range(h):
            pos_list.append((x, y))

    A_list = []
    for n in range(2, pic_num + 1):
        A_list = A_list + list(map(list, permutations(pos_list, n)))

    A_list_jump = []
    for j in range(len(A_list)):
        flag = True
        A = list(A_list[j])
        for k in range(len(A) - 1):
            movement = define_move(A[k], A[k + 1])
            if movement != 'jump':
                flag = False
                break
        if not flag:
            continue
        A_list_jump.append(A)

    B_list = list(map(list, product(range(B_size[1]), range(B_size[2]), range(B_size[3]))))
    labels = list(map(list, product(A_list_jump, B_list)))
    labels = list(map(add_target_pos, labels))
    return labels

def generate_neg_label(labels_pos, labels_all, num, min_neg_len):
    labels_neg = []
    while len(labels_neg)<num:
        index = random.randint(len(labels_all)-1)
        label_now = labels_all[index]
        if len(label_now[0]) >= min_neg_len and label_now not in labels_pos:
            labels_neg.append(label_now)
    return labels_neg

def generate_single_image(labels):
    agent_pos = labels[0]
    target_pos = labels[1]
    target_type = labels[2]
    background_type = labels[3]
    frame_type = labels[4]
    img = draw_mario_world(X=width, Y=height,
                           agent_x=agent_pos[0], agent_y=agent_pos[1],
                           target_x=target_pos[0], target_y=target_pos[1],
                           agent_icon=agents[0], target_icon=targets[target_type],
                           background_tile=backgrounds[background_type],
                           frame_tile=frames[frame_type])
    return img

def generate_images_by_label(labels):
    image_seq = []
    pos_info = labels[0]
    other_info = labels[1]
    for pos_now in pos_info:
        target_pos_now = other_info[0]
        target_type = other_info[1]
        background_type = other_info[2]
        frame_type = other_info[3]
        img_now = draw_mario_world(X=width, Y=height,
                                   agent_x=pos_now[0], agent_y=pos_now[1],
                                   target_x=target_pos_now[0], target_y=target_pos_now[1],
                                   agent_icon=agents[0], target_icon=targets[target_type],
                                   background_tile=backgrounds[background_type],
                                   frame_tile=frames[frame_type])
        image_seq.append(img_now)
    return image_seq

def generate_images_by_labels(labels, pre=False):
    images_in_cases = []
    if pre:
        for lbl in labels:
            images_in_cases.append(generate_single_image(lbl))
    else:
        for lbl in labels:
            images_in_cases.append(generate_images_by_label(lbl))
    return images_in_cases


if __name__ == '__main__':
    # =================== parameter set =====================
    random.seed(888)
    gen_pre = True
    p_pre = 0.01
    width = 3
    height = 3
    label_B_size = [width*height, 2, 3, 7]      # [target position, target type, background type, frame type]
    neg_num = 2000
    max_pos_num = 2000
    min_neg_num_default = 2
    path_p = 0.6        # the percentage of having path and terminate in neg case
    jump_p = 0.5        # the percentage of having jump and terminate path in neg case
    just_down_p = 0.2        # the percentage of all down_lowest path in neg case
    sea_path_p = 0.7
    right_priority_p = 0.2
    flower_path_p = 0.7
    # =================== folder set =====================
    folder = f'/home/jiny/AbdGen_data/dataset/'
    file = f'mario_right_priority_pretrain.npz'
    # file = f'mario.npz'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # =================== task set ===========================
    imgs_pos = {}
    lbls_pos = {}
    imgs_neg = {}
    lbls_neg = {}
    # task = ['left_priority','up_priority','just_down', 'right_priority','flower','sea', 'down_priority',
    #         'just_right', 'just_up', 'just_left', 'bomb_far','chess_jump',
    #         'right_one_step','up_one_step','left_one_step','down_one_step']#'far',
    task = ['right_priority']
    path_task = ['just_right','just_up','just_left',
                 'just_down','right_one_step','up_one_step',
                 'left_one_step','down_one_step']
    right_up_path_task = ['right_priority','up_priority']

    left_down_path_task = ['left_priority', 'down_priority']
    jump_task = ['chess_jump']
    sea_task = ['sea']
    flower_task = ['flower']

    for task_name in tqdm(task):
        label_neg = []
        min_neg_num = min_neg_num_default
        if task_name == 'right_priority':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = priority(width, height, label_B_size, 'right')
        elif task_name == 'left_priority':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = priority(width, height, label_B_size, 'left')
        elif task_name == 'up_priority':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = priority(width, height, label_B_size, 'up')
        elif task_name == 'down_priority':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = priority(width, height, label_B_size, 'down')
        elif task_name == 'just_right':
            max_pic_num = 3
            labels_fit_rule = just(width, height, label_B_size, 'right')
        elif task_name == 'just_left':
            max_pic_num = 3
            labels_fit_rule = just(width, height, label_B_size, 'left')
        elif task_name == 'just_up':
            max_pic_num = 3
            labels_fit_rule = just(width, height, label_B_size, 'up')
        elif task_name == 'just_down':
            max_pic_num = 3
            labels_fit_rule = just(width, height, label_B_size, 'down')
        elif task_name == 'right_one_step':
            max_pic_num = 2
            labels_fit_rule = onestep(width, height, label_B_size, 'right')
        elif task_name == 'left_one_step':
            max_pic_num = 2
            labels_fit_rule = onestep(width, height, label_B_size, 'left')
        elif task_name == 'up_one_step':
            max_pic_num = 2
            labels_fit_rule = onestep(width, height, label_B_size, 'up')
        elif task_name == 'down_one_step':
            max_pic_num = 2
            labels_fit_rule = onestep(width, height, label_B_size, 'down')
        elif task_name == 'sea':
            max_pic_num = 3
            labels_fit_rule = sea(width, height, label_B_size)
        elif task_name == 'far':
            max_pic_num = 2
            labels_fit_rule = far(width, height, label_B_size)
        elif task_name == 'bomb_far':
            max_pic_num = 2
            labels_fit_rule = bomb_far(width, height, label_B_size)
        elif task_name == 'chess_jump':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = chess_jump(width, height, max_pic_num, label_B_size)
        elif task_name == 'flower':
            max_pic_num = 5
            min_neg_num = 3
            labels_fit_rule = flower(width, height, label_B_size)
        all_labels = generate_all_labels(width, height, min(3,max_pic_num), label_B_size)
        if task_name in left_down_path_task:
            all_path_labels = generate_all_path_labels(width, height, max_pic_num, label_B_size, movement_list=['left','down'])
            label_neg = generate_neg_label(labels_fit_rule, all_path_labels, int(neg_num*path_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num-int(neg_num*path_p), min_neg_num)
        elif task_name in right_up_path_task:
            all_path_labels = generate_all_path_labels(width, height, max_pic_num, label_B_size, movement_list=['right','up'])
            label_neg = generate_neg_label(labels_fit_rule, all_path_labels, int(neg_num*path_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num-int(neg_num*path_p), min_neg_num)
        elif task_name in path_task:
            all_path_labels = generate_all_path_labels(width, height, max_pic_num, label_B_size, movement_list=['right', 'up','left','down'])
            label_neg = generate_neg_label(labels_fit_rule, all_path_labels, int(neg_num * path_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num - int(neg_num * path_p),min_neg_num)
        elif task_name in jump_task:
            all_jump_labels = generate_all_jump_labels(width, height, max_pic_num, label_B_size)
            label_neg = generate_neg_label(labels_fit_rule, all_jump_labels, int(neg_num*jump_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num-int(neg_num*jump_p), min_neg_num)
        elif task_name in sea_task:
            all_path_sea_labels = generate_all_path_labels(width, height, max_pic_num, label_B_size, background='sea')
            label_neg = generate_neg_label(labels_fit_rule, lbls_pos['just_down'], int(neg_num*just_down_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_path_sea_labels, int(neg_num*sea_path_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num-int(neg_num*just_down_p)-int(neg_num*sea_path_p), min_neg_num)
        elif task_name in flower_task:
            all_path_flower_labels = generate_all_path_labels(width, height, max_pic_num, label_B_size, background='flower')
            label_neg = generate_neg_label(labels_fit_rule, lbls_pos['right_priority'], int(neg_num*right_priority_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_path_flower_labels, int(neg_num*flower_path_p), min_neg_num)
            label_neg = label_neg + generate_neg_label(labels_fit_rule, all_labels, neg_num-int(neg_num*right_priority_p)-int(neg_num*flower_path_p), min_neg_num)
        else:
            label_neg = generate_neg_label(labels_fit_rule, all_labels, neg_num, min_neg_num)

        if len(labels_fit_rule)>max_pos_num:
            labels_fit_rule = rd.sample(labels_fit_rule, max_pos_num)


        if gen_pre:
            all_label_set = set()
            for i in range(len(labels_fit_rule)):
                temp = labels_fit_rule[i][1][:]
                if len(labels_fit_rule[i][0])>2:
                    for j in range(len(labels_fit_rule[i][0])):
                        pos = labels_fit_rule[i][0][j]
                        temp_2 = temp[:]
                        temp_2.insert(0, pos)
                        temp_2 = tuple(temp_2)
                        all_label_set.add(temp_2)

            for i in range(len(label_neg)):
                temp = label_neg[i][1][:]
                if len(label_neg[i][0])>2:
                    for j in range(len(label_neg[i][0])):
                        pos = label_neg[i][0][j]
                        temp_2 = temp[:]
                        temp_2.insert(0, pos)
                        temp_2 = tuple(temp_2)
                        all_label_set.add(temp_2)

            total_img_num = len(all_label_set)
            pre_img_num = int(total_img_num*p_pre)
            all_label_list = list(all_label_set)
            pre_label = rd.sample(all_label_list, pre_img_num)
            pre_img = generate_images_by_labels(pre_label,pre=True)
            np.savez(os.path.join(folder, file),
                     images_pre={task_name:pre_img},
                     labels_pre={task_name:pre_label})

        else:
            lbls_pos[task_name] = labels_fit_rule
            lbls_neg[task_name] = label_neg
            imgs_pos[task_name] = generate_images_by_labels(labels_fit_rule)
            imgs_neg[task_name] = generate_images_by_labels(label_neg)

            # test
            # index_img = 0
            # for imgs in imgs_pos[task[0]][800]:
            #     test = Image.fromarray(imgs)
            #     test.save("./test"+str(index_img)+".jpeg")
            #     index_img+=1

            np.savez(os.path.join(folder, file),
                     images_pos=imgs_pos,
                     images_neg=imgs_neg,
                     labels_pos=lbls_pos,
                     labels_neg=lbls_neg)

