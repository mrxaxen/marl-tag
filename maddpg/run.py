import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3
import time

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def run():
    # parallel_env = simple_speaker_listener_v4.parallel_env(
            # continuous_actions=True, render_mode="human")


    EVAL_INTERVAL = 1000
    MAX_STEPS = 800_000
    evaluate_performance = False
    best_score = 0
    total_steps = 0
    episode = 0
    eval_scores = []
    eval_steps = []

    render_mode = "human" if evaluate_performance is True else "none"

    parallel_env = simple_tag_v3.parallel_env(num_good=1, num_adversaries=3,
                                              num_obstacles=0, continuous_actions=True, render_mode=render_mode)
    _, _ = parallel_env.reset()
    n_agents = parallel_env.max_num_agents

    # print(n_agents)

    actor_dims = []
    n_actions = []
    for agent in parallel_env.agents:
        # print(f'agent is: {agent}, actor dims is: {parallel_env.observation_space(agent).shape[0]}')
        actor_dims.append(parallel_env.observation_space(agent).shape[0])
        n_actions.append(parallel_env.action_space(agent).shape[0])
        # print(f'agent is: {agent}, actions is: {parallel_env.action_space(agent).shape[0]}')
    critic_dims = sum(actor_dims) + sum(n_actions) # 3*12 + 10 + 4*5
    # print(f'critic dims is: {critic_dims}')
    # print(f"actor_dims is: {actor_dims}")
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           env=parallel_env, gamma=0.95, alpha=1e-4, beta=1e-3)
    critic_dims = sum(actor_dims)
    memory = MultiAgentReplayBuffer(5_000_000, critic_dims, actor_dims, # changed to 5m
                                    n_actions, n_agents, batch_size=1024)


    score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
    eval_scores.append(score)
    eval_steps.append(total_steps)

    if evaluate_performance:
        maddpg_agents.load_checkpoint()

    while total_steps < MAX_STEPS:
        obs, _ = parallel_env.reset()
        terminal = [False] * n_agents
        # score = 0
        while not any(terminal):
            # if evaluate_performance:
                # time.sleep(0.25)  # to slow down the action

            actions = maddpg_agents.choose_action(obs)

            obs_, reward, done, trunc, info = parallel_env.step(actions)

            list_done = list(done.values())
            list_obs = list(obs.values())
            list_reward = list(reward.values())
            list_actions = list(actions.values())
            list_obs_ = list(obs_.values())
            list_trunc = list(trunc.values())

            state = obs_list_to_state_vector(list_obs)
            state_ = obs_list_to_state_vector(list_obs_)

            terminal = [d or t for d, t in zip(list_done, list_trunc)]
            memory.store_transition(list_obs, state, list_actions, list_reward,
                                    list_obs_, state_, terminal)

            if total_steps % 100 == 0 and not evaluate_performance:
                maddpg_agents.learn(memory)
            obs = obs_
            total_steps += 1

        if total_steps % EVAL_INTERVAL == 0:
            score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
            eval_scores.append(score)
            eval_steps.append(total_steps)

        if not evaluate_performance:
            if score > best_score:
                print("saving checkpoint")
                maddpg_agents.save_checkpoint()
                best_score = score


        episode += 1

    np.save('data/maddpg_scores.npy', np.array(eval_scores))
    np.save('data/maddpg_steps.npy', np.array(eval_steps))


def evaluate(agents, env, ep, step, n_eval=3):
    score_history = []
    for i in range(n_eval):
        obs, _ = env.reset()
        score = 0
        terminal = [False] * env.max_num_agents
        while not any(terminal):
            actions = agents.choose_action(obs, evaluate=True)
            obs_, reward, done, trunc, info = env.step(actions)

            list_trunc = list(trunc.values())
            list_reward = list(reward.values())
            list_done = list(done.values())

            terminal = [d or t for d, t in zip(list_done, list_trunc)]

            obs = obs_
            score += sum(list_reward)
        score_history.append(score)
    avg_score = np.mean(score_history)


    print(f'Evaluation episode {ep} train steps {step}'
          f' average score {avg_score:.1f}')
    return avg_score


if __name__ == '__main__':
    run()
    # evaluate()
