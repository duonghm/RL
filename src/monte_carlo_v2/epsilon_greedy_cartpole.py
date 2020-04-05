from gym.envs import toy_text
from collections import defaultdict
from collections import namedtuple
from typing import List
import numpy as np
import gym
import monte_carlo.plotting as plotting

Step = namedtuple('Step', ('state', 'action', 'reward'))


MIN_MAX_POSITION = (-2.4, 2.4)
NUM_STATE_POSITION = 20
MIN_MAX_VELOCITY = (-3, 3)
NUM_STATE_VELOCITY = 30
MIN_MAX_ANGLE = (-0.418, 0.418)
NUM_STATE_ANGLE = 30
MIN_MAX_VELOCITY_AT_TIP = (-2, 2)
NUM_STATE_VELOCITY_AT_TIP = 30


def discrete_value(value, min_v, max_v, num_state):
    if value <= min_v:
        return 0
    elif value >= max_v:
        return num_state - 1
    state = int((value - min_v) / (max_v - min_v) * num_state)
    return state


def obs_to_state(observation):
    position, velocity, angle, velocity_at_tip = observation
    # position = round(position, 1)
    # velocity = round(velocity, 1)
    # angle = round(angle, 1)
    # velocity_at_tip = round(velocity_at_tip, 1)
    # return str([position, velocity, angle, velocity_at_tip])
    position = discrete_value(position, MIN_MAX_POSITION[0], MIN_MAX_POSITION[1], NUM_STATE_POSITION)
    velocity = discrete_value(velocity, MIN_MAX_VELOCITY[0], MIN_MAX_VELOCITY[1], NUM_STATE_VELOCITY)
    angle = discrete_value(angle, MIN_MAX_ANGLE[0], MIN_MAX_ANGLE[1], NUM_STATE_ANGLE)
    velocity_at_tip = discrete_value(velocity_at_tip, MIN_MAX_VELOCITY_AT_TIP[0],
                                     MIN_MAX_VELOCITY_AT_TIP[1], NUM_STATE_VELOCITY_AT_TIP)
    state_idx = [position, velocity, angle, velocity_at_tip]
    state_space = [NUM_STATE_POSITION, NUM_STATE_VELOCITY, NUM_STATE_ANGLE, NUM_STATE_VELOCITY_AT_TIP]
    return np.ravel_multi_index(state_idx, state_space)


class Policy(object):
    def __init__(self, env, epsilon, q):
        self.env = env
        self.epsilon = epsilon
        self.q = q

    def choose_action(self, state):
        def calc_prob(action, epsilon, a_size, best_action):
            if action == best_action:
                return 1 - epsilon + epsilon / a_size
            else:
                return epsilon / a_size

        q = self.q
        if len(q[state]) > 0:
            best_action = max(q[state], key=q[state].get)
            probs_dict = {a: calc_prob(a, self.epsilon, self.env.action_space.n, best_action)
                          for a in range(self.env.action_space.n)}
            return np.random.choice(list(probs_dict.keys()), p=list(probs_dict.values()))
        else:
            return self.env.action_space.sample()


def create_episode(env, policy, is_render=False) -> List[Step]:
    ep = []
    obs_i = env.reset()
    while True:
        if is_render:
            env.render()
        state = obs_to_state(obs_i)
        a_i = policy.choose_action(state)
        obs_i_1, r_i_1, done, _ = env.step(a_i)
        ep.append(Step(state, a_i, r_i_1))
        obs_i = obs_i_1
        if done:
            break
    return ep


def mc_control(env, num_episodes, gamma, epsilon):
    q = defaultdict(lambda: defaultdict(float))
    policy = Policy(env, epsilon, q)
    returns = defaultdict(list)

    for i in range(num_episodes):
        if i % 100 == 0:
            is_render = True
        else:
            is_render = False
        ep = create_episode(env, policy, is_render)
        print('ep %s: %s' % (i, len(ep)))
        g = 0
        visited_sa = set()
        for step in ep[::-1]:
            g = gamma * g + step.reward
            current_sa = (step.state, step.action)
            if current_sa not in visited_sa:
                returns[current_sa].append(g)
                q[step.state][step.action] = np.average(returns[current_sa])
                policy.q = q
            visited_sa.add(current_sa)
    return policy


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    print("Training ....")
    policy = mc_control(env, 10000, 0.99, 0.05)
    env.close()
