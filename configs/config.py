import torch

class Config:
    def __init__(self):
        self.action_type = 'Discrete'   # 'Discrete' 'Continuous'
        self.actor_hidden_dim = 64
        self.actor_activation = 'ReLu'  # 'ReLu' 'LeakReLu'  'Tanh'
        self.actor_weight_init = 'orthogonal' # 'orthogonal' 'xavier_normal' 'normal' 'constant'
        self.actor_optimizer = 'Adam'  # 'Adam' 'RMSprop'
        self.actor_lr = 3e-4

        self.critic_hidden_dim = 64
        self.critic_activation = 'ReLu' # 'ReLu' 'LeakReLu'  'Tanh'
        self.critic_weight_init = 'orthogonal'  # 'orthogonal' 'xavier_normal' 'normal' 'constant'
        self.critic_optimizer = 'Adam'  # 'Adam' 'RMSprop'
        self.critic_lr = 1e-3

        self.adaptive_lr = False
        if self.adaptive_lr:
            self.actor_lr_scheduler = 'LambdaLR'  # 'LambdaLR' 'StepLR' 'ExponentialLR' 'ReduceLROnPlateau()'
            self.critic_lr_scheduler = 'LambdaLr' # 'LambdaLR' 'StepLR' 'ExponentialLR' 'ReduceLROnPlateau()'

        self.clip_grad_norm = 10
        self.ppo_update_times = 10
        self.sur_epsilon = 0.25


        self.gamma = 0.99

        # training random seed
        self.seed = 1
        # training network parameter update cycle
        self.net_update_cycle = 10
        # result log cycle
        self.log_cycle = 50
        # max epochs
        self.max_training_epochs = 3000
        # max steps per epoch
        self.max_steps_per_epoch = 1000

        # save path
        self.save_path = 'D:/zhangjie/zj_program_code/RL/RL_Algorithm/results'
        self.save_model_cycle = int(self.max_training_epochs / 10)

        self.use_gpu = True
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                print('GPU is not available on this computer device')
                self.device = 'cpu'
        else:
            self.device = 'cpu'

class Env_config:
    def __init__(self):
        self.action_dim = 4
        self.state_dim = 8
        self.env_name = 'CartPole-v0'
