from typing import Any, List, Sequence

from lbforaging.foraging import ForagingEnv, Agent
from lbforaging.foraging.environment import Action

from .constants import AGENT_REGISTRY


class AHTLBFEnv:
    _env: ForagingEnv
    obss: Any
    npc_agents: List[Agent]

    def __init__(self, *args, **kwargs):
        self._env = ForagingEnv(*args, **kwargs)
        self.npc_agents = []
        self.obss = None
        self.render_mode = "human"

    @property
    def observation_space(self):
        return self._env.observation_space[0]

    @property
    def action_space(self):
        return self._env.action_space[0]

    def create_aht_agent(self, agent_type: str):
        assert agent_type in AGENT_REGISTRY, f"Unknown agent type {agent_type}"
        return AGENT_REGISTRY[agent_type](self.players[0])

    def set_npc_agents(self, npc_agent_types: List[str]):
        assert len(npc_agent_types) == self._env.n_agents - 1
        self.npc_agents = []
        for i, agent_type in enumerate(npc_agent_types):
            assert agent_type in AGENT_REGISTRY, f"Unknown agent type {agent_type}"
            self.npc_agents.append(AGENT_REGISTRY[agent_type](self.players[i + 1]))

    def step(self, action: int):
        actions = (
            [agent.act(self.obss[m + 1]) for m, agent in enumerate(self.npc_agents)]
            if self.obss
            else [Action.NONE] * len(self.npc_agents)
        )

        outs = self._env.step([action] + actions)

        assert (
            "full_observations" in outs[-1]
        ), "info dict should contain full_observations key"
        self.obss = outs[-1]["full_observations"]

        return tuple(x[0] if isinstance(x, Sequence) else x for x in outs)

    def reset(self):
        return self._env.reset()[0]

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    def seed(self, seed=None):
        return self._env.seed(seed)

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        return getattr(self._env, item)
