import numpy as np
import gym
import scipy
import scipy.signal
import matplotlib.pyplot as plt
from functools import partial


# Simple Tabular Monte Carlo Control with Epsilon Greedy policy
# as described very closely in Chapter 5 in
#
# Sutton, R., & Barto, A. (1998). Reinforcement learning, an introduction.
# Cambridge: MIT Press/Bradford Books.

def TabularEpsilonGreedyPolicy(Q, eps, state):
    sample = np.random.random_sample()

    num_actions = Q.shape[1]

    if sample > eps:
        max_val = Q[state, :].max()
        max_indices = np.where(np.abs(Q[state, :] - max_val) < 1e-5)[0]
        rand_idx = np.random.randint(len(max_indices))
        max_action = max_indices[rand_idx]

        return max_action
    else:
        return np.random.randint(num_actions)


def discretize_val(val, min_val, max_val, num_states):
    """
    Discretizes a single float
    if val < min_val, it gets a discrete value of 0
    if val >= max_val, it gets a discrete value of num_states-1

    Args:
        val (float): value to discretize
        min_val (float): lower bound of discretization
        max_val (float): upper bound of discretization
        num_states (int): number of discrete states

    Returns:
        float: discrete value
    """
    state = int(num_states * (val - min_val) / (max_val - min_val))
    if state >= num_states:
        state = num_states - 1
    if state < 0:
        state = 0
    return state


def obs_to_state(num_states, lower_bounds, upper_bounds, obs):
    """
    Turns an observation in R^N, into a discrete state

    Args:
        num_states (list): list of number of states for each dimension of observation
        lower_bounds (list): list of lowerbounds for discretization
        upper_bounds (list): list of upperbounds for discretization
        obs (list): observation in R^N to discretize

    Returns:
        int: discrete state
    """
    state_idx = []
    for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
        state_idx.append(discretize_val(ob, lower, upper, num))

    return np.ravel_multi_index(state_idx, num_states)


def get_discounted_rewards(rewards, gamma):
    """
    Gets Discounted rewards at every timestep

    Args:
        rewards (numpy array): a list of [r_1, r_2, ...]
        where r_i = r(x_i, u_i)
        gamma (float): discount factor

    Returns:
        numpy array: a list of [R_1, R_2, ...]
        where R_i = \sum_{n=i} r_n * gamma^{n-i}
    """
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]


def rollout(env, policy, get_state, max_iter=10000, render=False):
    """
    Simulates one episode of the environment following a policy

    Args:
        env (TYPE): openai gym env
        policy (function): function that takes in state and returns an action
        get_state (TYPE): function to get state from observation
        max_iter (int, optional): maximum number of iterations to simulate
        render (bool, optional): True if you want to render the environment

    Returns:
        TYPE: Description
    """
    obs = env.reset()
    rewards = []
    actions = []
    states = []
    for _ in range(max_iter):
        state = get_state(obs)
        action = policy(state)

        actions.append(action)
        states.append(state)

        if (render):
            env.render()
            time.sleep(0.01)
        [obs, reward, done, info] = env.step(action)
        rewards.append(reward)

        if done:
            break

    return [states, actions, rewards]


# tabular monte carlo
class TabularMC(object):
    def __init__(self, num_states, num_actions, gamma=1, eps=0.1, eps_decay=0.9999, first_visit=True):
        self.num_states = num_states
        self.num_actions = num_actions
        self.first_visit = first_visit

        self.gamma = gamma
        self.start_eps = eps
        self.eps_decay = eps_decay

        self.reset_policy()

    def reset_policy(self):
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q_num = np.zeros((self.num_states, self.num_actions))
        self.visited_num = np.zeros((self.num_states, self.num_actions))
        self.eps = self.start_eps

    def curr_policy(self, copy=False):
        if copy:
            return partial(TabularEpsilonGreedyPolicy, np.copy(self.Q), self.eps)
        else:
            return partial(TabularEpsilonGreedyPolicy, self.Q, self.eps)

    def update(self, env, get_state):
        policy = self.curr_policy()
        [states, actions, rewards] = rollout(env, policy, get_state)

        updated_states = set()

        discounted_rewards = get_discounted_rewards(np.array(rewards), self.gamma)
        for state, action, disc_rew in zip(states, actions, discounted_rewards):

            if self.first_visit:
                if state in updated_states:
                    continue

                updated_states.add(state)

            # incremental averaging
            self.Q_num[state, action] += 1
            self.Q[state, action] += (disc_rew - self.Q[state, action]) / self.Q_num[state, action]
            self.visited_num[state, action] += 1

        self.eps *= self.eps_decay

        return np.sum(np.array(rewards))


if __name__ == '__main__':
    class RunningAverage(object):
        def __init__(self, N):
            self.N = N
            self.vals = []
            self.num_filled = 0

        def push(self, val):
            if self.num_filled == self.N:
                self.vals.pop(0)
                self.vals.append(val)
            else:
                self.vals.append(val)
                self.num_filled += 1

        def get(self):
            return float(sum(self.vals)) / self.num_filled


    env = gym.make('CartPole-v0')
    # env = gym.wrappers.Monitor(env, '/tmp/cartpole', force=True)
    num_actions = env.action_space.n
    num_states = [1, 8, 8, 8]
    lower_bounds = [-4.8, -3, -0.418, -2]
    upper_bounds = [4.8, 3, 0.418, 2]

    get_state = partial(obs_to_state, num_states, lower_bounds, upper_bounds)

    state_len = np.prod(np.array(num_states))
    mc = TabularMC(state_len, num_actions, first_visit=False, eps_decay=0.99)

    num_episodes = 1000

    rewards = np.zeros(num_episodes)
    avg_calc = RunningAverage(100)
    for i in range(num_episodes):
        total_reward = mc.update(env, get_state)
        avg_calc.push(total_reward)

        rewards[i] = avg_calc.get()
        print("Iteration " + str(i) + " reward: " + str(total_reward))

    env.close()

    plt.plot(rewards)

    plt.title('rewards over episodes')
    plt.xlabel('episodes')
    plt.ylabel('reward')

    plt.show()
