import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from configs.config import Env_config, Config
from agent import Agent


env_config = Env_config()
env_name = env_config.env_name
env = gym.make(env_name).unwrapped
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env_config.state_dim = state_dim
env_config.action_dim = action_dim

config = Config()
random_seed = config.seed
torch.manual_seed(random_seed)
env.seed(random_seed)

AGENT = Agent(config, env_config)


EP_R = []
log_running_reward = 0
log_running_episodes = 0



max_epochs = config.max_training_epochs
max_step_per_epoch = config.max_steps_per_epoch
for i_epoch in range(1, max_epochs + 1):
    state = env.reset()
    current_ep_reward = 0

    for i_step in range(max_step_per_epoch):
        action, log_prob = AGENT.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        AGENT.store_tuples(state, action, log_prob, reward, done)
        state = next_state
        current_ep_reward += reward
        #env.render()

        if done:
            break
    log_running_reward += current_ep_reward
    log_running_episodes += 1

    if i_epoch % config.net_update_cycle == 0:
        AGENT.alg.update()

    if i_epoch % config.log_cycle == 0:
        log_avg_reward = log_running_reward / log_running_episodes
        log_avg_reward = round(log_avg_reward, 4)
        print('epoch', i_epoch, 'avg_reward', log_avg_reward)
        EP_R.append(log_avg_reward)
        AGENT.alg.writer.add_scalar('Epoch_Reward', log_avg_reward, i_epoch)
        log_running_reward = 0
        log_running_episodes = 0

    if i_epoch % config.save_model_cycle == 0:
        AGENT.alg.save()
        print("============================================================================================")
        print('save model')
