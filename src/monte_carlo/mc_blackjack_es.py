from gym.envs import toy_text
from collections import defaultdict
from collections import namedtuple
from typing import List
import numpy as np
import monte_carlo.plotting as plotting

Step = namedtuple('Step', ('state', 'action', 'reward'))


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))


env = toy_text.BlackjackEnv()
# env.seed(0)


def create_episode(env, policy) -> List[Step]:
    ep = []
    s0 = env.reset()
    a0 = env.action_space.sample()
    # print('s0: %s, a0: %s' % (s0, a0))
    state, reward, done, _ = env.step(a0)
    ep.append(Step(s0, a0, reward))
    while not done:
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        ep.append(Step(state, action, reward))
        state = next_state
    return ep


def find_best_action(q_action_dict):
    return max(q_action_dict.keys(), key=lambda k: q_action_dict[k])


def mc_control(env, num_episodes, discount_factor):
    policy = defaultdict(int)  # TODO: Fix me
    q = defaultdict(lambda: defaultdict(float))
    returns = defaultdict(list)

    for i in range(num_episodes):
        ep = create_episode(env, policy)
        g = 0
        visited_lst = []
        for step in ep[::-1]:
            g = discount_factor * g + step.reward
            state_action = (step.state, step.action)
            if state_action not in visited_lst:
                returns[state_action].append(g)
                # q[state_action] = np.average(returns[state_action])
                q[step.state][step.action] = np.average(returns[state_action])
                policy[step.state] = find_best_action(q[step.state])
            visited_lst.append(state_action)
    return q, policy


def simulate_episode(env, policy):
    ep = []
    state = env.reset()
    for t in range(100):
        action = policy[state]
        next_state, reward, done, info = env.step(action)
        ep.append(Step(state, action, reward))  # (S_i, A_i, R_(i+1))
        state = next_state
        if done:
            if reward == 1:
                result = 'win'
            elif reward == 0:
                result = 'draw'
            else:
                result = 'lose'
            ep.append((state, toy_text.blackjack.score(env.dealer), result))
            break
    return ep


def check_win_ratio(env, policy, num_ep):
    sim_results = [simulate_episode(env, policy) for i in range(num_ep)]
    total_wins = sum(1 if x[-1][-1]=='win' else 0 for x in sim_results)
    total_draw = sum(1 if x[-1][-1] == 'draw' else 0 for x in sim_results)
    total_lose = sum(1 if x[-1][-1] == 'lose' else 0 for x in sim_results)
    win_ratio = total_wins / num_ep
    draw_ratio = total_draw / num_ep
    lose_ratio = total_lose / num_ep
    return win_ratio, draw_ratio, lose_ratio


if __name__ == '__main__':
    q_10000, policy_10000 = mc_control(env, 10000, 0.99)
    print(check_win_ratio(env, policy_10000, 10000))
    V = defaultdict(float)
    for state, action_value in q_10000.items():
        print(state, action_value, max(action_value.values()))
        V[state] = max(action_value.values())
    plotting.plot_value_function(V, title="Optimal Value Function")
