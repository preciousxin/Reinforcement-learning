import numpy as np
import torch
from algs.ppo import PPO


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Agent:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.buffer = RolloutBuffer()
        self.alg = PPO(self.config, self.env, self.buffer)


    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.config.device)
            action, log_prob = self.alg.actor_old(state)

        return action.item(), log_prob.item()

    def store_tuples(self, state, action, log_prob, reward, is_terminal):
        state = torch.tensor(state, dtype=torch.float32).to(self.config.device)
        action = torch.tensor(action ,dtype=torch.int).to(self.config.device)
        log_prob = torch.tensor(log_prob, dtype=torch.float32).to(self.config.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.config.device)
        is_terminal = torch.tensor(is_terminal, dtype=torch.bool).to(self.config.device)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)
