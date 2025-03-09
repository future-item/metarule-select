config = {'dataset': 'mario',
          'seed': 42,
          'GPU': 1,
          'num_episodes': 500,
          'num_trails_per_episode': 8,
          'actor_lr': 1e-5,     # Initial learning rate
          'lr_step_size':50,
          'lr_gamma':0.8,      # for step_lr
          'n_hiddens':256,
          'num_heads':8,        # n_hiddens is a multiple of num_heads
          'epochs':8,      # Training rounds of a sequence
          'eps': 0.2,       # Parameters of the truncated range in PPO
          'epsilon': 0.3,      # Initial Exploration rate
          'epsilon_decay_rate': 0.9995,
          'test_p': 0.3,
          'dropout': 0.1,
          'pl_time_limit': 20,
          'error_reward': 0,
          'num_action_sample': 32,
          'noise_rate': 0.3,
          'nn_restorer': True,
          'hopfield': True,
          'noise': True,
          'relu': False,
          'padding_mask': True,
          'gae_mode':'self',
          'data_path': '/home/jiny/AbdGen_data/dataset/mario.npz',
          'excel_result_path': './result_hopfield_noise_0.3_add_train/result.xlsx',
          'result_path': './result_hopfield_noise_0.3_add_train/',
          'pl_path':'./prolog_hopfield_noise_0.3_add_train/',
          'model_path':'./model_hopfield_noise_0.3_add_train/',
          'log_file_path':'./result_hopfield_noise_0.3_add_train/log.txt',
          'len_guarantee_task': ['right_priority','up_priority','left_priority','down_priority','flower'],
          'all_task':['right_priority','up_priority', 'just_left','just_down',
                      'right_one_step','left_one_step','chess_jump','bomb_far', 'flower',
                      'left_priority', 'down_priority', 'just_right', 'just_up',
                      'up_one_step', 'down_one_step'],
          'train_task':['right_priority', 'up_priority', 'left_priority',
                        'just_left', 'just_down', 'just_up',
                        'right_one_step','left_one_step','down_one_step',
                        'chess_jump','bomb_far', 'flower'],
          'test_task':['down_priority', 'just_right',
                       'up_one_step'],
          'max_pic_num': 5,
          'B_len': 3,
          'num_pos': [5, 10, 20],
          'num_neg': [15, 20, 30, 40, 50],
          'max_pos':20,
          'max_neg':50,
          'num_metarule': 6,
          # 'metarule_encoding_map':{
          #     0:[1,4,5,2,4,5,0,0,0],
          #     1:[1,4,5,2,5,4,0,0,0],
          #     2:[1,4,5,2,4,3,4,5,0],
          #     3:[1,4,5,2,4,5,3,5,0],
          #     4:[1,4,5,2,4,6,3,6,5],
          #     5:[1,4,5,2,4,6,1,6,5]},
          # 'metarule_encoding_map': {
          #     0: [1, 0, 0, 0, 0, 0],
          #     1: [0, 1, 0, 0, 0, 0],
          #     2: [0, 0, 1, 0, 0, 0],
          #     3: [0, 0, 0, 1, 0, 0],
          #     4: [0, 0, 0, 0, 1, 0],
          #     5: [0, 0, 0, 0, 0, 1]},
          'metarule_embedding_dim': 16,
          'metarule_pl_map': {
              0: "metarule([P,Q], [P,A,B], [[Q,A,B]]).\n",
              1: "metarule([P,Q], [P,A,B], [[Q,B,A]]).\n",
              2: "metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).\n",
              3: "metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).\n",
              4: "metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]).\n",
              5: "metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).\n"},
          'grd_map': {
                      (0, 0): 0,
                      (1, 0): 1,
                      (2, 0): 2,
                      (0, 1): 3,
                      (1, 1): 4,
                      (2, 1): 5,
                      (0, 2): 6,
                      (1, 2): 7,
                      (2, 2): 8},
          'label_pl_map': {
                        -1: None,
                        0: '[0,0]',
                        1: '[1,0]',
                        2: '[2,0]',
                        3: '[0,1]',
                        4: '[1,1]',
                        5: '[2,1]',
                        6: '[0,2]',
                        7: '[1,2]',
                        8: '[2,2]'}
}


bk_prepare = [":- use_module('metagol').\n:- style_check(-singleton).\n\nwidth(2).\nheight(2).\n\nmetagol:min_clauses(1).\nmetagol:max_clauses(4).\n\n",
              "body_pred(up/2).\nbody_pred(down/2).\nbody_pred(left/2).\nbody_pred(right/2).\nbody_pred(bomb/1).\nbody_pred(sea/1).\nbody_pred(chess/1).\nbody_pred(flower/1).\nbody_pred(far/2).\nbody_pred(jump/2).\nbody_pred(terminate/2).\n\n",
              "up([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-\n    height(H),\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    X1 is X0,\n    Y0 < H,\n    Y1 is Y0 + 1.\n\n",
              "down([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    X1 is X0,\n    Y0 > 0,\n    Y1 is Y0 - 1.\n\n",
              "left([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    Y1 is Y0,\n    X0 > 0,\n    X1 is X0 - 1.\n\n",
              "right([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-\n    width(W),\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    Y1 is Y0,\n    X0 < W,\n    X1 is X0 + 1.\n\n",
              "jump([[X0,Y0],[X1,Y1]|T], [[X1,Y1]|T]) :-\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    abs(X1-X0) =:= 1,\n    abs(Y1-Y0) =:= 1.\n\n",
              "terminate([[X0,Y0]], [[X1,Y1]|T]) :-\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    X1 is X0,\n    Y1 is Y0.\n\n",
              "flower([A,B,C]) :-\n    number(C),\n    C is 0.\n\n",
              "sea([A,B,C]) :-\n    number(C),\n    C is 1.\n\n",
              "chess([A,B,C]) :-\n    number(C),\n    C is 2.\n\n",
              "bomb([A,B,C]) :-\n    number(C),\n    B is 1.\n\n",
              "far([[X0,Y0],[X1,Y1]|_], [[X2,Y2]|T]) :-\n    number(X0),\n    number(X1),\n    number(Y0),\n    number(Y1),\n    number(X2),\n    number(Y2),\n    abs(X2-X1)+abs(Y2-Y1) > abs(X2-X0)+abs(Y2-Y0).\n\n"]























