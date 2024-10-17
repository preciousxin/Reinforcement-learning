import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNet(nn.Module):
    def __init__(self, config, env):
        super(Actor, self).__init__()
        self.config = config
        self.env = env
        self.action_dim = self.env.action_dim
        if self.config.actor_activation == 'ReLu':
            self.act_func = nn.ReLU()
        elif self.config.actor_activation == 'LeakReLu':
            self.act_func = nn.LeakyReLU()
        elif self.config.actor_activation == 'Tanh':
            self.act_func = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(self.env.state_dim, self.config.actor_hidden_dim),
            self.act_func,
            nn.Linear(self.config.actor_hidden_dim, self.config.actor_hidden_dim),
            self.act_func,
            nn.Linear(self.config.actor_hidden_dim, self.action_dim * 2))

        # weight initialization
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                if self.config.actor_weight_init == 'orthogonal':
                    weight_init = nn.init.orthogonal_
                weight_init(param, gain = 1)


    def forward(self, inputs):
        x = self.net(inputs)

        # continuous action case
        if self.config.action_type == 'Continuous':
            a_mu = x[:,:self.action_dim]
            a_logstd = x[:,self.action_dim:]
            a_std = a_logstd.exp()
            return a_mu, a_std

    def get_action(self, inputs):
        a_mu, a_std = self.forward(inputs)
        dist = Normal(a_mu, a_std)
        u = dist.sample()
        action = torch.tanh(u)
        action = action.detach().cpu().numpy().item()

        return action

    def evaluate(self, inputs, epsilon = 1e-6):
        a_mu, a_std = self.forward(inputs)
        noise = torch.randn_like(a_mu, requires_grad = True).to(self.config.device)

        u = a_mu + noise * a_std
        action = torch.tanh(u)

        # calculate the log(\pi(u|s))  (action before tanh())
        log_pi = -0.5*((u-a_mu)/a_std).pow(2)-a_std.log()-math.log(math.sqrt(2*math.pi))
        # same as below:
        # log_pi = -0.5*noise.pow(2)-a_std.log()-math.log(math.sqrt(2*math.pi))
        # same as below:
        # dist = Normal(a_mu, a_std)
        # log_pi = dist.log_prob(u)

        # calculate the log(\pi(a|s)) (action after tanh() Enforcing Action Bounds)
        log_pi = log_pi - torch.log(1-action.pow(2)+epsilon)
        return action, log_pi

class CriticNet(nn.Module):
    def __init__(self, config, env):
        super(CriticNet, self).__init__()
        self.config = config
        self.env = env
        self.action_dim = self.env.action_dim

        if self.config.critic_activation == 'ReLu':
            self.act_func = nn.ReLU()
        elif self.config.critic_activation == 'LeakReLu':
            self.act_func = nn.LeakyReLU()
        elif self.config.critic_activation == 'Tanh':
            self.act_func = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(self.env.state_dim+self.env.action_dim , self.config.critic_hidden_dim),
            self.act_func,
            nn.Linear(self.config.critic_hidden_dim, self.config.critic_hidden_dim),
            self.act_func,
            nn.Linear(self.config.critic_hidden_dim, 1))

        # 权重参数初始化
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                if self.config.critic_weight_init == 'orthogonal':
                    weight_init = nn.init.orthogonal_
                weight_init(param, gain=1)

    def forward(self, inputs_s, inputs_a):
        # inputs_s is the input of state
        # inputs_a is the input of action
        # for the continous action case 
        inputs = torch.cat([inputs_s, inputs_a], dim=-1)
        q_value = self.net(inputs)
        return q_value
            
        
        
class SAC:
    def __init__(self, config, env, buffer):
        self.alg_name = 'SAC'
        self.config = config
        self.env = env
        self.buffer = buffer

        self.actor = ActorNet(self.config, self.env).to(self.config.device)
        self.critic = CriticNet(self.config, self.env).to(self.config.device)

        if self.config.actor_optimizer == 'Adam':
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.config.actor_lr)
        elif self.config.actor_optimizer == 'RMSprop':
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr = self.config.actor_lr)

        if self.config.critic_optimizer == 'Adam':
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.config.critic_lr)
        elif self.config.critic_optimizer == 'RMSprop':
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr = self.config.critic_lr)


        # 'LambdaLR' 'StepLR' 'ExponentialLR' 'ReduceLROnPlateau()'
        if self.config.adaptive_lr:
            if self.config.actor_lr_scheduler == 'LambdaLR':
                lambda_lr = lambda epoch: 0.9 ** epoch
                self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda_lr)
                self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda_lr)
            elif self.config.actor_lr_scheduler == 'ExponentialLR':
                gamma_lr = 0.5
                self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=gamma_lr)
                self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=gamma_lr)

        self.MSELoss = nn.MSELoss()

    def update(self):
        

    
