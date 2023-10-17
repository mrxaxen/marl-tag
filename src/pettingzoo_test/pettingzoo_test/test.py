from os import rename
from typing import List
import rclpy
from rclpy.context import Context
from rclpy.node import Node
from pettingzoo.mpe import simple_tag_v3

from rclpy.parameter import Parameter

class PettingzooNode(Node):

    def __init__(self, seed) -> None:
        super().__init__('pettingzoo_node')
        self.env = simple_tag_v3.env(render_mode="human")
        self.env.reset(seed=seed)
    

def main():
    rclpy.init()
    pettingZooNode = PettingzooNode(seed=42)
    
    while True:
        for agent in pettingZooNode.env.agent_iter():
            observation, reward, termination, truncation, info = pettingZooNode.env.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = pettingZooNode.env.action_space(agent).sample()

            pettingZooNode.env.step(action)
        pettingZooNode.env.close()


if __name__ == "__main__":
    main()