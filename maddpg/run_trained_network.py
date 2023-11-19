import numpy as np
import torch as T
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3
from networks import ActorNetwork

import threading
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


EVAL_INTERVAL = 1000
MAX_STEPS = 40000
evaluate_performance = True
best_score = 0
total_steps = 0
episode = 0
eval_scores = []
eval_steps = []



eval_env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, max_cycles=40000, continuous_actions=True,
                                              num_obstacles=0, render_mode="none")
env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3, max_cycles=40000, continuous_actions=True,
                                              num_obstacles=0, render_mode="human")

env.metadata["render_fps"] = 60
eval_env.metadata["render_fps"] = 60

eval_env.reset(seed=42)
env.reset(seed=42)
n_agents = env.max_num_agents

"""
    def __init__(self, actor_dims, critic_dims, n_actions,
                 n_agents, agent_idx, chkpt_dir, min_action,
                 max_action, alpha=1e-4, beta=1e-3, fc1=64,
                 fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        agent_name = 'agent_%s' % agent_idx
        self.agent_idx = agent_idx
        self.min_action = min_action
        self.max_action = max_action

        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir,
                                  name=agent_name+'_actor')
"""
"""
actor_dims = []
n_actions = []
for agent in parallel_env.agents:
    actor_dims.append(parallel_env.observation_space(agent).shape[0])
    n_actions.append(parallel_env.action_space(agent).shape[0])
"""
# load up the 4 networks
fc1 = 64
fc2 = 64
adversary_dims = 12
agent_dims = 10
n_actions = 5
alpha = 1e-4
device = 'cuda:0' if T.cuda.is_available() else 'cpu'

adversary_0_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                                  name="agent_0"+'_target_actor')
adversary_1_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                                  name="agent_1"+'_target_actor')
adversary_2_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                                  name="agent_2"+'_target_actor')
agent_0_network = ActorNetwork(alpha, agent_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                                  name="agent_3"+'_target_actor')
# print(next(agent_0_network.parameters()).is_cuda)
# agent_0_network = agent_0_network.to(device)

# Parallel
playback_actions = []
max_steps = 2000
def play_env():
    stepcount = 0
    observations, infos = eval_env.reset()
    run = True
    while eval_env.agents and run:
        actions = {
            "adversary_0": adversary_0_network(T.tensor(observations["adversary_0"], device=device)).detach().to("cpu"),
            "adversary_1": adversary_1_network(T.tensor(observations["adversary_1"], device=device)).detach().to("cpu"),
            "adversary_2": adversary_2_network(T.tensor(observations["adversary_2"], device=device)).detach().to("cpu"),
            "agent_0": agent_0_network(T.tensor(observations["agent_0"], device=device), ).detach().to("cpu"),
        }
        playback_actions.append(actions)
        observations, rewards, terminations, truncations, infos = eval_env.step(actions)
        print(f"Eval step: {stepcount}")
        if stepcount >= max_steps:
            run = False
        stepcount += 1


play_env()
stepcount = 0
run = True
print(playback_actions)
while env.agents and run:
    actions = playback_actions.pop(0)
    env.step(actions)
    stepcount += 1
    print(f"Step: {stepcount}")
    if stepcount >= max_steps:
        run = False
env.close()

"""
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    t_obs = T.tensor(observation)
    action = None
    # time.sleep(0.01)  # to slow down the action
    # print(f'obs is: {t_obs}')
    t_obs = t_obs.to(device)
    # print(t_obs.is_cuda)

    if agent == "agent_0":

        possible_actions = agent_0_network(t_obs)
        possible_actions = possible_actions.to("cpu")
        action = np.argmax(possible_actions.detach().numpy())
        # action = 0
        # print(possible_actions)
        # print(max_action)
    elif agent == "adversary_0":
        possible_actions = adversary_0_network(t_obs)
        possible_actions = possible_actions.to("cpu")
        action = np.argmax(possible_actions.detach().numpy())
        # action = 0
    elif agent == "adversary_1":
        possible_actions = adversary_1_network(t_obs)
        possible_actions = possible_actions.to("cpu")
        action = np.argmax(possible_actions.detach().numpy())
        # action = 0
    elif agent == "adversary_2":
        possible_actions = adversary_2_network(t_obs)
        possible_actions = possible_actions.to("cpu")
        action = np.argmax(possible_actions.detach().numpy())
        # action = 0
    if termination or truncation:
        action = None

    env.step(action)
env.close()
"""
