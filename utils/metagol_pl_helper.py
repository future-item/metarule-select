from config import config_mario, config_mnist
import asyncio

def case_writer_mario(label):
    max_pic_num = config_mario.config['max_pic_num']
    label_pl_map = config_mario.config['label_pl_map']
    case_str = 'f(['
    A_str_list = []
    B_str_list = []
    # the last label "frame type" is not writen in the pl file
    for i in range(len(label)-1):
        if i < max_pic_num:
            if label_pl_map[label[i]] is not None:
                A_str_list.append(label_pl_map[label[i]])
        elif i == max_pic_num:
            B_str_list.append(label_pl_map[label[i]])
        else:
            B_str_list.append(str(label[i]))
    A_str = ','.join(A_str_list)
    B_str = ','.join(B_str_list)
    case_str = case_str + A_str + '],[' + B_str + '])'
    return case_str

def case_writer_mnist(label, task_name):
    reverse_task = config_mnist.config['reverse_task']
    A_str_list = []
    B_str_list = []

    for i in range(len(label)):
        if task_name in reverse_task:
            if i == 0:
                A_str_list.append(str(label[i]))
            else:
                if label[i] != -1:
                    B_str_list.append(str(label[i]))
        else:
            if i == len(label)-1:
                B_str_list.append(str(label[i]))
            else:
                if label[i] != -1:
                    A_str_list.append(str(label[i]))
    A_str = ','.join(A_str_list)
    B_str = ','.join(B_str_list)
    if task_name in reverse_task:
        case_str = 'f('
        case_str = case_str + A_str + ',[' + B_str + '])'
    else:
        case_str = 'f(['
        case_str = case_str + A_str + '],' + B_str + ')'
    return case_str

def task_writer_mario(task_name, pos_label, neg_label):
    file_name = config_mario.config['pl_path']+ 'task_' + task_name + '.pl'
    with open(file_name, 'w') as f:
        f.write(":- ['task" + "_" + task_name + "_bk.pl'].\n")
        f.write("\n")

        pos_list = []
        pos_str = "a :- Pos=["
        for j in range(len(pos_label)):
            pos_list.append(case_writer_mario(pos_label[j]))
        pos_str = pos_str + ',\n          '.join(pos_list) + '],\n'

        neg_list = []
        neg_str = "     Neg=["
        for j in range(len(neg_label)):
            neg_list.append(case_writer_mario(neg_label[j]))
        neg_str = neg_str + ',\n          '.join(neg_list) + '],\n'
        f.write(pos_str+neg_str+'     learn(Pos,Neg).')

def task_writer_mnist(task_name, pos_label, neg_label):
    file_name = config_mnist.config['pl_path']+ 'task_' + task_name + '.pl'
    with open(file_name, 'w') as f:
        f.write(":- ['task" + "_" + task_name + "_bk.pl'].\n")
        f.write("\n")

        pos_list = []
        pos_str = "a :- Pos=["
        for j in range(len(pos_label)):
            pos_list.append(case_writer_mnist(pos_label[j], task_name))
        pos_str = pos_str + ',\n          '.join(pos_list) + '],\n'

        neg_list = []
        neg_str = "     Neg=["
        for j in range(len(neg_label)):
            neg_list.append(case_writer_mnist(neg_label[j], task_name))
        neg_str = neg_str + ',\n          '.join(neg_list) + '],\n'
        f.write(pos_str+neg_str+'     learn(Pos,Neg).')

def bk_writer_mario(task_name, metarule_statu):
    file_name = config_mario.config['pl_path'] + 'task_' + task_name + '_bk.pl'
    num_metarule = config_mario.config['num_metarule']
    metarule_pl_map = config_mario.config['metarule_pl_map']

    with open(file_name, 'w') as f:
        f.write(config_mario.bk_prepare[0])
        for i in range(num_metarule):
            if metarule_statu[i] == 1.0:
                f.write(metarule_pl_map[i])
        f.write("\n")
        for i in config_mario.bk_prepare[1:]:
            f.write(i)

def bk_writer_mnist(task_name, metarule_statu):
    file_name = config_mnist.config['pl_path'] + 'task_' + task_name + '_bk.pl'
    num_metarule = config_mnist.config['num_metarule']
    metarule_pl_map = config_mnist.config['metarule_pl_map']

    with open(file_name, 'w') as f:
        f.write(config_mnist.bk_prepare[0])
        for i in range(num_metarule):
            if metarule_statu[i] == 1.0:
                f.write(metarule_pl_map[i])
        f.write("\n")
        if task_name in config_mnist.config['add_multi_task']:
            f.write(config_mnist.body_pred[0])
            f.write(config_mnist.body_pred[1])
            f.write(config_mnist.body_pred[2])
        else:
            f.write(config_mnist.body_pred[3])
            f.write(config_mnist.body_pred[4])
            f.write(config_mnist.body_pred[5])
        for i in config_mnist.bk_prepare[1:]:
            f.write(i)

async def run_pl(file_path, time):
    cmd = "/usr/bin/swipl --stack-limit=8g -s {} -g a -t halt".format(file_path)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=time)
        if proc.returncode == 0:
            return 0, stdout.decode('UTF-8')
        else:
            return -1, stderr.decode('UTF-8')
    except asyncio.TimeoutError as e:
        if proc.returncode is None:
            try:
                proc.kill()
            except OSError:
                # Ignore 'no such process' error
                pass
        return -2, "Timeout " + str(e)  # timeout error


def read_pl_out(pl_out_str, need_len=False):
    prog_list = []
    for line in pl_out_str.splitlines():
        if line[0] == 'f':
             prog_list.append(line)
    prog_str = '\n'.join(prog_list)
    if need_len:
        return prog_str, len(prog_list)
    else:
        return prog_str


