config = {'dataset': 'mnist',
          'seed': 42,
          'GPU': 1,
          'num_episodes': 150,
          'num_trails_per_episode': 8,
          'actor_lr': 1e-5,     # Initial learning rate
          'lr_step_size':50,
          'lr_gamma':0.8,      # for step_lr
          'n_hiddens':256,
          'num_heads':8,        # n_hiddens is a multiple of num_heads
          'epochs':8,      # Training rounds of a sequence
          'eps': 0.2,       # Parameters of the truncated range in PPO
          'epsilon': 0.5,      # Initial Exploration rate
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
          'data_path': '/home/jiny/AbdGen_data/dataset/mnist.npz',
          'excel_result_path': './mnist_result_hopfield_noise_0.3/result.xlsx',
          'result_path': './mnist_result_hopfield_noise_0.3/',
          'pl_path':'./mnist_prolog_hopfield_noise_0.3/',
          'model_path':'./mnist_model_hopfield_noise_0.3/',
          'log_file_path':'./mnist_result_hopfield_noise_0.3/log.txt',
          'all_task':['decreasing_sequence', 'increasing_sequence', 'add_priority',
                      'multi_priority', 'cumulative_sum', 'cumulative_product',
                      'reverse_cumulative_sum', 'reverse_cumulative_product'],
          'add_multi_task':['add_priority', 'multi_priority', 'cumulative_sum',
                            'cumulative_product','reverse_cumulative_sum',
                            'reverse_cumulative_product'],
          'reverse_task' : ['reverse_cumulative_sum', 'reverse_cumulative_product','decreasing_sequence'],
          'max_pic_num': 5,
          'B_len': 1,
          'num_pos': [5, 10, 20],
          'num_neg': [20, 30, 40],
          'max_pos':20,
          'max_neg':40,
          'num_metarule': 6,
          'metarule_embedding_dim': 16,
          'metarule_pl_map': {
              0: "metarule([P,Q], [P,A,B], [[Q,A,B]]).\n",
              1: "metarule([P,Q], [P,A,B], [[Q,B,A]]).\n",
              2: "metarule([P,Q,R], [P,A,B], [[Q,A],[R,A,B]]).\n",
              3: "metarule([P,Q,R], [P,A,B], [[Q,A,B],[R,B]]).\n",
              4: "metarule([P,Q,R], [P,A,B], [[Q,A,C],[R,C,B]]).\n",
              5: "metarule([P,Q], [P,A,B], [[Q,A,C],[P,C,B]]).\n"},
}


bk_prepare = [":- use_module('metagol').\n:- use_module(library(clpfd)).\n\n:- style_check(-singleton).\n\nmetagol:min_clauses(1).\nmetagol:max_clauses(4).\n\n",
              "add([A,B|T], [C|T]) :-\n    number(A),\n    number(B),\n    C #= A+B.\n\n",
              "multi([A,B|T], [C|T]) :-\n    number(A),\n    number(B),\n    C #= A*B.\n\n",
              "eq([A], B) :-\n    number(A),\n    number(B),\n    A #= B.\n\n",
              "less([A,B|T], [B|T]) :-\n    number(A),\n    number(B),\n    A #< B.\n\n",
              "less([A, B], C) :-\n    number(A),\n    number(B),\n    A #< B.\n\n",
              "more([A,B|T], [B|T]) :-\n    number(A),\n    number(B),\n    A #> B.\n\n",
              "more([A, B], C) :-\n    number(A),\n    number(B),\n    A #> B.\n\n",
              "zero(A) :-\n    number(A),\n    A #= 0.\n\n"]

body_pred = ['body_pred(add/2).\n',
             'body_pred(multi/2).\n',
             'body_pred(eq/2).\n',
             'body_pred(less/2).\n',
             'body_pred(more/2).\n',
             'body_pred(zero/1).\n']























