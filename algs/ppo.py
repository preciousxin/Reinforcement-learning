import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import os



class Actor(nn.Module):
    def __init__(self, config, env):
        super(Actor, self).__init__()
        self.config = config
        self.env = env
        self.fc1 = nn.Linear(self.env.state_dim, self.config.actor_hidden_dim)
        self.fc2 = nn.Linear(self.config.actor_hidden_dim, self.config.actor_hidden_dim)
        self.fc3 = nn.Linear(self.config.actor_hidden_dim, self.env.action_dim)

        
    def forward(self, inputs):
        if self.config.actor_activation == 'ReLu':
            self.act_func = F.relu
        elif self.config.actor_activation == 'LeakReLu':
            self.act_func = F.leaky_relu
        elif self.config.actor_activation == 'Tanh':
            self.act_func = F.tanh
        x = self.act_func(self.fc1(inputs))
        x = self.act_func(self.fc2(x))
        action_probs = F.softmax(self.fc3(x))

        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

class ActorNet(nn.Module):
    def __init__(self, config, env):
        super(ActorNet, self).__init__()
        self.config = config
        self.env = env

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
            nn.Linear(self.config.actor_hidden_dim, self.env.action_dim),
            nn.Softmax())

        # 权重参数初始化
        for name, param in self.net.named_parameters():
            if 'weight' in name:
                if self.config.actor_weight_init == 'orthogonal':
                    weight_init = nn.init.orthogonal_
                weight_init(param, gain = 1)

    def forward(self, inputs):
        x = self.net(inputs)
        dist = Categorical(x)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate(self, state_inputs, action_inputs):
        x = self.net(state_inputs)
        dist = Categorical(x)
        log_probs = dist.log_prob(action_inputs)
        dist_entropy = dist.entropy()

        return log_probs, dist_entropy


class CriticNet(nn.Module):
    def __init__(self, config, env):
        super(CriticNet, self).__init__()
        self.config = config
        self.env = env

        if self.config.critic_activation == 'ReLu':
            self.act_func = nn.ReLU()
        elif self.config.critic_activation == 'LeakReLu':
            self.act_func = nn.LeakyReLU()
        elif self.config.critic_activation == 'Tanh':
            self.act_func = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(self.env.state_dim, self.config.critic_hidden_dim),
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

    def forward(self, inputs):
        x = self.net(inputs)
        return x

class PPO:
    def __init__(self, config, env, buffer):
        self.alg_name = 'PPO'
        self.config = config
        self.env = env
        self.buffer = buffer
        self.writer = SummaryWriter('{}/{}/logs'.format(self.config.save_path, self.alg_name))
        self.update_count = 0
        self.save_path = self.config.save_path + '/' + self.alg_name

        self.actor = ActorNet(self.config, self.env).to(self.config.device)
        self.critic = CriticNet(self.config, self.env).to(self.config.device)

        self.actor_old = ActorNet(self.config, self.env).to(self.config.device)
        self.critic_old = CriticNet(self.config, self.env).to(self.config.device)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

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
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.config.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.config.device)
        rewards = rewards / (rewards.std() + 1e-7)

        states_old = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.config.device)
        actions_old = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.config.device)
        log_probs_old = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(self.config.device)

        for _ in range(self.config.ppo_update_times):
            log_probs_new, dist_entropy = self.actor.evaluate(states_old, actions_old)
            state_values = self.critic(states_old)
            state_values = torch.squeeze(state_values)
            delta = rewards - state_values
            advantages = delta.detach()

            ratios = torch.exp(log_probs_new - log_probs_old)

            sur1 = ratios * advantages
            sur2 = ratios * torch.clamp(ratios, 1-self.config.sur_epsilon, 1+self.config.sur_epsilon)

            actor_loss = -torch.min(sur1, sur2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.clip_grad_norm)
            self.actor_optimizer.step()


            critic_loss = self.MSELoss(rewards, state_values) #- 0.01 * dist_entropy
            self.critic_optimizer.zero_grad()
            critic_loss.mean().backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.clip_grad_norm)
            self.critic_optimizer.step()

            if self.config.adaptive_lr:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        self.buffer.clear()
        self.update_count += 1
        self.writer.add_scalar('loss/actor_loss', actor_loss, self.update_count)
        self.writer.add_scalar('loss/critic_loss', critic_loss.mean(), self.update_count)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def save(self):
        model_path = self.save_path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = model_path + '/' + 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        actor_model_path = model_path + '/' + 'actor_net'
        if not os.path.exists(actor_model_path):
            os.makedirs(actor_model_path)
        current_num_pretrained_files = next(os.walk(actor_model_path))[2]
        run_num_pretrained = len(current_num_pretrained_files)
        checkpoint_actor_path = actor_model_path + '/' + '{}_{}.pth'.format(self.env.env_name, run_num_pretrained)
        torch.save(self.actor_old.state_dict(), checkpoint_actor_path)

        critic_model_path = model_path + '/' + 'critic_net'
        if not os.path.exists(critic_model_path):
            os.makedirs(critic_model_path)
        current_num_pretrained_files = next(os.walk(critic_model_path))[2]
        run_num_pretrained = len(current_num_pretrained_files)
        checkpoint_critic_path = critic_model_path + '/' + '{}_{}.pth'.format(self.env.env_name, run_num_pretrained)
        torch.save(self.critic_old.state_dict(), checkpoint_critic_path)

    def load(self):
        pass


