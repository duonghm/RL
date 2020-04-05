import gym
import numpy as np
import time
import pickle
from collections import namedtuple, defaultdict


Step = namedtuple('Step', ('state', 'action', 'reward'))


def find_best_action(q_action_dict):
    best_action = max(q_action_dict.keys(), key=lambda k: q_action_dict[k])
    # print(q_action_dict, best_action)
    return best_action


def norm_state(state):
    normed = [round(state[0], 2), round(state[1], 2), round(state[2], 2), round(state[3], 2)]
    # print(str(normed))
    return str(normed)


def generate_ep(env, policy):
    ep = []
    s0 = env.reset()
    a0 = env.action_space.sample()
    state, reward, done, _ = env.step(a0)
    ep.append(Step(norm_state(s0), a0, reward))
    for i in range(200):
        if norm_state(state) not in policy:
            action = env.action_space.sample()
        else:
            action = policy[norm_state(state)]
        next_state, reward, done, _ = env.step(action)
        ep.append(Step(norm_state(state), action, reward))
        state = next_state
        if done:
            break
    return ep


def mc_control(env, num_ep, gamma):
    policy = defaultdict(int)
    q = defaultdict(lambda: defaultdict(float))
    returns_sum = defaultdict(float)
    returns_cnt = defaultdict(float)

    for i in range(num_ep):
        ep = generate_ep(env, policy)
        print(len(ep), ep)

        sa_in_ep = set([(step.state, step.action) for step in ep])
        for state, action in sa_in_ep:
            sa_pair = (state, action)
            first_idx = next(i for i, step in enumerate(ep) if step.state == state and step.action == action)
            g = sum([
                step.reward * (gamma**i) for i, step in enumerate(ep[first_idx:])
            ])
            returns_sum[sa_pair] += g
            returns_cnt[sa_pair] += 1.0
            q[state][action] = returns_sum[sa_pair] / returns_cnt[sa_pair]

    return policy, q


def check_policy(env, policy):
    observation = env.reset()
    for t in range(10000):
        env.render()
        hit_or_miss = 'hit' if norm_state(observation) in policy else 'miss'
        action = policy[norm_state(observation)]
        print(norm_state(observation), action, hit_or_miss)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    start_time = time.time()
    policy, _, = mc_control(env, int(1e4), 0.99)
    print('Duration: %s min' % ((time.time() - start_time) / 60))
    # pickle.dump(policy, '1e5.policy')
    check_policy(env, policy)
    env.close()