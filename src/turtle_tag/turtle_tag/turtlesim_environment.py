
from typing import TypeVar
from gymnasium import Space
import numpy as np

from pettingzoo import ParallelEnv

ObsType = TypeVar("ObsType")
AgentID = TypeVar("AgentID")
ActionType = TypeVar("ActionType")


class TurtlesimEnv(ParallelEnv):

    def __init__(self):
        """
        TODO: implement
        """
        raise NotImplemented

    # Parent interface
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """
        TODO: implement
        """
        raise NotImplemented

    # Parent interface
    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """
        TODO: implement
        """
        raise NotImplemented

    # Parent interface
    def render(self) -> None | np.ndarray | str | list:
        """
        TODO: start turtlesim
        """
        raise NotImplemented

    # Parent interface
    def close(self):
        """
        Close turtlesim
        """

    # Parent interface
    def state(self) -> np.ndarray:
        """
        TODO: implement
        """

    # Parent interface
    def observation_space(self, agent: AgentID) -> Space:
        """
        TODO: implement
        """

    # Parent interface
    def action_space(self, agent: AgentID) -> Space:
        """
        TODO: implement
        """


class TurtlesimScenario:

    def make_world(self):
        """
        Create elements of the world
        TODO: implement
        """
        raise NotImplemented

    def reset_world(self, world, np_random):
        """
        Create the initial conditions of the world.
        TODO: implement
        """
        raise NotImplemented
