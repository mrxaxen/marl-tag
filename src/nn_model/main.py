import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_tag_v3
import time

def obs_list_to_state_vector(observation):  # might need to tweak
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'simple_tag'

    # env = make_env(scenario)
    num_good = 1
    num_adversaries = 3
    num_obstacles = 0

    PRINT_INTERVAL = 500
    N_GAMES = 10000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    env = simple_tag_v3.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles,
                                     max_cycles=N_GAMES, continuous_actions=False, render_mode="none")
    # render mode human to see

    env.reset(seed=42)

    n_agents = num_good + num_adversaries
    actor_dims = []

    for i in range(num_good):
        # size of observation space of good agent is: num other adv * 2 + num_obs * 2 + 4 (self_vel & pos)
        # actor_dims.append(num_adversaries * 2 + num_obstacles * 2 + 4) # maybe it should be half size??
        actor_dims.append(num_adversaries * 2 + num_obstacles * 2 + 4 + 2) # trying with +2 size, to get around matmult error

    for i in range(num_adversaries):
        actor_dims.append((num_adversaries - 1) * 2 + num_obstacles * 2 + num_good * 4 + 4)
    critic_dims = sum(actor_dims)

    """
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    """

    # action space is a list of arrays, assume each agent has same action space
    # n_actions = env.action_space[0].n

    n_actions = 5 # move 4 ways or dont move
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)


    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        # obs = env.reset()
        obs, info = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            # if evaluate: # have way to only render when done
                # env.render()
                #time.sleep(0.1) # to slow down the action for the video

            # need to transform obs into a nested array instead of dict
            obs = list(obs.values())
            # print(f'obs is: {obs}')
            obs[-1] = np.append(obs[-1], [0., 0.])
            # print(f'obs is: {obs}')
            # print(np.shape(obs))
            obs = np.array(obs, dtype=np.float32) # dtype object to not throw warning

            # print(f'obs is: {obs}')
            # print(f'obs[-1] is: {obs[-1]}')
            # obs[-1] = np.append([0., 0.], obs[-1])
            # print(f'obs is: {obs}')
            obs = np.vstack(obs)
            # print(f'post vstack obs is: {obs}')
            # print(f'obs_ is : {obs}')
            # obs = obs.astype(float)
            # print(f'obs_ is : {obs}')
            actions = maddpg_agents.choose_action(obs) # gets raw observation
            # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow!

            # obs_, reward, done, info = env.step(actions)
            # print(f'actions are: {actions}')
            # pettingzoo wants: actions are: {'adversary_0': 1, 'adversary_1': 3, 'adversary_2': 1, 'agent_0': 1}
            # print(f'max first action is: {np.argmax(actions[0])}')
            actions_list = [np.argmax(x) for x in actions]
            # print(f'actions are: {actions_list}')

            actions = { # find way to adapt to multiple
                "adversary_0": actions_list[0],
                "adversary_1": actions_list[1],
                "adversary_2": actions_list[2],
                "agent_0": actions_list[-1]
            }

            obs_, reward, done, truncations, info = env.step(actions) # truncations, info not used?


            # print(f'rewards are: {reward}')
            reward = list(reward.values())
            # print(f'rewards are: {reward}')
            reward = [float(i) for i in reward]
            # print(f'rewards are: {reward}')

            obs_ = list(obs_.values())
            obs_[-1] = np.append(obs_[-1], [0., 0.])
            obs_ = np.array(obs_, dtype=np.float32)  # dtype object to not throw warning
            obs_ = np.vstack(obs_)
            # print(f'obs_ is : {obs_}')
            # obs_ = obs_.astype(float)
            # print(f'obs_ is : {obs_}')

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            actions_list = [float(i) for i in actions_list]

            memory.store_transition(obs, state, actions_list, reward, obs_, state_, done) # changed to store actions list

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                # maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
    env.close()