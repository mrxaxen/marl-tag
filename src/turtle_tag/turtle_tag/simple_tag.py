from typing import Any, Callable
from pettingzoo.mpe import simple_tag_v3

import rclpy
from rclpy.node import Node

from numpy.typing import NDArray


class SimpleTagService(Node):

    def __init__(self, seed: int, policy: Callable[[dict[str, NDArray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]], int] = None):
        super().__init__('simple_tag_service')

        self.env = simple_tag_v3.env(render_mode="human")
        self.provided_policy = policy
        self.run_sim(seed)

    def run_sim(self, seed):
        self.env.reset(seed=seed)

        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            action = None

            if termination or truncation:
                action = None
            else:
                action = self.run_policy(agent=agent)

            self.env.step(action)
            print(f"Observations:{observation}")
            print(f"Rewards:{reward}")
            print(f"Terminations:{termination}")
            print(f"Truncations:{truncation}")
            print(f"Infos:{info}")

        #self.env.close()
    '''
    def run_sim_parallel(self):
        observations, infos = self.env.reset()
        
        while self.env.agents:
            actions = self.run_policy(observations)
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.env.close()

        print(f"Observations:{observations}")
        print(f"Rewards:{rewards}")
        print(f"Terminations:{terminations}")
        print(f"Truncations:{truncations}")
        print(f"Infos:{infos}")
    '''

    def run_policy(self, observation: NDArray = None, agent: Any = None):
        if self.provided_policy is None or observation is None:
            return self.env.action_space(agent).sample()
        else:
            return self.provided_policy(observation)

    def run_policy_parallel(self, observations: dict[str, NDArray]):
        if self.provided_policy is None:
            return {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
        else:
            return self.provided_policy(observations)


def main():
    rclpy.init()

    simple_tag_service = SimpleTagService(42)

    rclpy.spin(simple_tag_service)

    rclpy.shutdown()
