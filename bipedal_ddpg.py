import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time


from ddpg_agent import Agent

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)


def ddpg(n_episodes=10000, max_t=700):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf

    for i_episode in range(1, n_episodes+1):
    
        state = env.reset()
        agent.reset()
        score = 0
    

        for t in range(max_t):
            action = agent.act(state)[0]
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward


            env.render()

            if done:
                break 

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")

        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   

    return scores

# scores = ddpg()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.savefig('graph.png')
# plt.show()



# To run on pretrained model
pretrained_dict = torch.load('checkpoint_actor.pth')
model_dict = agent.actor_local.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
agent.actor_local.load_state_dict(pretrained_dict)



pretrained_dict = torch.load('checkpoint_critic.pth')
model_dict = agent.critic_local.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
agent.critic_local.load_state_dict(pretrained_dict)


state = env.reset()
agent.reset()   

while True:

    action = agent.act(state)[0]
    env.render()
    # time.sleep(0.1)
    next_state, reward, done, _ = (env.step(action))
    state = next_state

    if done:
        break
        
env.env.close()