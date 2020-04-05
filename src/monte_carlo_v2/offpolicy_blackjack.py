import gym
from gym.envs import toy_text
from collections import defaultdict
from collections import namedtuple


Step = namedtuple('Step', ('state', 'action', 'reward'))


class McOffPolicyControl(object):
    def __init__(self, env, is_check_training=True):
        self.env = env
        self.q = defaultdict(lambda: defaultdict(float))
        self.c = defaultdict(lambda: defaultdict(float))
        self.target_policy = defaultdict(int)
        self.is_check_training = is_check_training

    def do_action(self, state):
        action = self.target_policy.get(state)
        if action is None:
            return self.env.action_space.sample()
        return action

    def obs_to_state(self, obs):
        return obs

    def behaviour_policy(self, state):
        return self.env.action_space.sample()

    def _generate_ep(self, policy):
        ep = []
        obs_i = self.env.reset()
        s_i = self.obs_to_state(obs_i)
        while True:
            a_i = policy(s_i)
            obs_i_1, r_i_1, done, _ = self.env.step(a_i)
            ep.append(Step(s_i, a_i, r_i_1))
            s_i = self.obs_to_state(obs_i_1)
            if done:
                break
        return ep

    def generate_behaviour_ep(self):
        return self._generate_ep(self.behaviour_policy)

    def generate_target_ep(self):
        return self._generate_ep(self.do_action)

    def check_win_ratio(self, num_ep):
        sim_results = [self.generate_target_ep() for i in range(num_ep)]
        total_wins = sum(1 if x[-1].reward == 1 else 0 for x in sim_results)
        total_draw = sum(1 if x[-1].reward == 0 else 0 for x in sim_results)
        total_lose = sum(1 if x[-1].reward == -1 else 0 for x in sim_results)
        win_ratio = total_wins / num_ep
        draw_ratio = total_draw / num_ep
        lose_ratio = total_lose / num_ep
        return win_ratio, draw_ratio, lose_ratio

    def train(self, num_ep, gamma):
        for i in range(num_ep):
            ep = self.generate_behaviour_ep()
            g = 0
            w = 1
            for step in ep[::-1]:
                g = gamma * g + step.reward
                c = self.c[step.state][step.action]
                c += w
                q = self.q[step.state][step.action]
                q = q + w / c * (g - q)
                self.c[step.state][step.action] = c
                self.q[step.state][step.action] = q
                best_action = max(self.q[step.state], key=self.q[step.state].get)
                self.target_policy[step.state] = best_action
                if step.action != best_action:
                    continue
                w = w * 1 / self.env.action_space.n
            if i % 100 == 0:
                print("Step %i: %s" %(i, self.check_win_ratio(10000)))


if __name__ == '__main__':
    env = toy_text.BlackjackEnv()
    mc_control = McOffPolicyControl(env)
    mc_control.train(10000, 0.9)

