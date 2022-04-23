from typing import List
from collections import defaultdict, namedtuple
import numpy as np
import gym


Step = namedtuple('Step', ('state', 'action', 'reward', 'next_state'))


class EpsGreedyPolicy(object):
    def __init__(self,
                 eps: float,
                 actions: List):
        self.eps = eps
        self.actions = actions

    def select_action(self, state, q):
        q_s = q[state]
        if len(q_s) > 0:
            best_action = max(q_s, key=q_s.get)
            part_prob = self.eps / len(self.actions)
            select_prob = [
                1 - self.eps + part_prob if a == best_action else part_prob
                for a in self.actions
            ]
            return np.random.choice(self.actions, p=select_prob)
        else:
            return np.random.choice(self.actions)


class QTabularTrainer(object):
    def __init__(self,
                 env,
                 policy: EpsGreedyPolicy,
                 alpha: float,
                 gamma: float
                 ):
        self.env = env
        self.Q = defaultdict(lambda: defaultdict(float))
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

    def obs_to_state(self, obs):
        # TODO: Implement this
        raise NotImplementedError()

    def train(self, n_episode):
        for i in range(n_episode):
            obs = self.env.reset()
            s = self.obs_to_state(obs)
            while True:
                a = self.policy.select_action(s, self.Q)
                obs_1, r, done, _ = self.env.step(a)
                s1 = self.obs_to_state(obs_1)
                self.Q[s][a] = self.Q[s][a] + self.alpha * (
                        r + self.gamma * max(self.Q[s1].values()) - self.Q[s][a]
                )
                s = s1
                if done:
                    break
