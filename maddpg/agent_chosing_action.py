import numpy as np
import torch as T
from agent import Agent
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3
from networks import ActorNetwork
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


EVAL_INTERVAL = 1000
MAX_STEPS = 4000
evaluate_performance = True
best_score = 0
total_steps = 0
episode = 0
eval_scores = []
eval_steps = []
num_adversaries = 3
num_good = 1
num_agents = num_good + num_adversaries

env = simple_tag_v3.env(num_good=num_good, num_adversaries=num_adversaries, max_cycles=MAX_STEPS, continuous_actions=False,
                                              num_obstacles=0, render_mode="human")
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
critic_dims = agent_dims + num_adversaries*adversary_dims + num_agents*n_actions
alpha = 1e-4
device = 'cuda:0' if T.cuda.is_available() else 'cpu'

# min_action = env.action_space('agent_0').low
# max_action = env.action_space('agent_0').high

min_action = 0
max_action = 4

"""
adversary_0_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                                  name="agent_0"+'_target_actor')
adversary_1_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_1_actor",
                                  name="agent_1"+'_target_actor')
adversary_2_network = ActorNetwork(alpha, adversary_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_2_actor",
                                  name="agent_2"+'_target_actor')
agent_0_network = ActorNetwork(alpha, agent_dims, fc1, fc2, n_actions,
                                  chkpt_dir="/tmp/maddpg/simple_tag/agent_3_actor",
                                  name="agent_3"+'_target_actor')
"""

adversary_0_agent = Agent(actor_dims=adversary_dims, critic_dims=critic_dims,
                               n_actions=5, n_agents=num_agents, agent_idx=0,
                               alpha=1e-4, beta=1e-3, tau=0.01, fc1=fc1,
                               fc2=fc2, chkpt_dir="/tmp/maddpg/simple_tag/agent_0_actor",
                               gamma=0.95, min_action=min_action,
                               max_action=max_action)

adversary_1_agent = Agent(actor_dims=adversary_dims, critic_dims=critic_dims,
                               n_actions=5, n_agents=num_agents, agent_idx=1,
                               alpha=1e-4, beta=1e-3, tau=0.01, fc1=fc1,
                               fc2=fc2, chkpt_dir="/tmp/maddpg/simple_tag/agent_1_actor",
                               gamma=0.95, min_action=min_action,
                               max_action=max_action)

adversary_2_agent = Agent(actor_dims=adversary_dims, critic_dims=critic_dims,
                               n_actions=5, n_agents=num_agents, agent_idx=2,
                               alpha=1e-4, beta=1e-3, tau=0.01, fc1=fc1,
                               fc2=fc2, chkpt_dir="/tmp/maddpg/simple_tag/agent_2_actor",
                               gamma=0.95, min_action=min_action,
                               max_action=max_action)

agent_0_agent = Agent(actor_dims=agent_dims, critic_dims=critic_dims,
                               n_actions=5, n_agents=num_agents, agent_idx=2,
                               alpha=1e-4, beta=1e-3, tau=0.01, fc1=fc1,
                               fc2=fc2, chkpt_dir="/tmp/maddpg/simple_tag/agent_3_actor",
                               gamma=0.95, min_action=min_action,
                               max_action=max_action)



# print(next(agent_0_network.parameters()).is_cuda)
# agent_0_network = agent_0_network.to(device)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    t_obs = T.tensor(observation)
    action = None
    # time.sleep(0.01)  # to slow down the action
    # print(f'obs is: {t_obs}')
    t_obs = t_obs.to(device)
    # print(t_obs.is_cuda)

    if agent == "agent_0":

        # possible_actions = agent_0_network(t_obs)
        # possible_actions = possible_actions.to("cpu")
        # action = np.argmax(possible_actions.detach().numpy())

        agent_0_action = agent_0_agent.choose_action(observation=t_obs, evaluate=True)
        # print(f'agent_0 obs is: {t_obs}')
        # print(f'agent_0 action is: {agent_0_action}')
        action = np.argmax(agent_0_action)

        # action = 0
        # print(possible_actions)
        # print(max_action)
    elif agent == "adversary_0":
        # possible_actions = adversary_0_network(t_obs)
        # possible_actions = possible_actions.to("cpu")
        # action = np.argmax(possible_actions.detach().numpy())

        adversary_0_action = adversary_0_agent.choose_action(observation=t_obs, evaluate=True)

        # print(f'adversary_0 action is: {adversary_0_action}')
        action = np.argmax(adversary_0_action)

        # action = 0
    elif agent == "adversary_1":
        # possible_actions = adversary_1_network(t_obs)
        # possible_actions = possible_actions.to("cpu")
        # action = np.argmax(possible_actions.detach().numpy())
        # action = 0

        adversary_1_action = adversary_1_agent.choose_action(observation=t_obs, evaluate=True)

        # print(f'adversary_1 action is: {adversary_1_action}')
        action = np.argmax(adversary_1_action)

    elif agent == "adversary_2":
        # possible_actions = adversary_2_network(t_obs)
        # possible_actions = possible_actions.to("cpu")
        # action = np.argmax(possible_actions.detach().numpy())
        # action = 0

        adversary_2_action = adversary_2_agent.choose_action(observation=t_obs, evaluate=True)

        # print(f'adversary_2 action is: {adversary_2_action}')
        action = np.argmax(adversary_2_action)

    if termination or truncation:
        action = None

    env.step(action)
env.close()