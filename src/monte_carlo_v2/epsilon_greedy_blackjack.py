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


def create_episode(env, policy) -> List[Step]:
    ep = []
    s_i = env.reset()
    while True:
        a_i = policy.choose_action(s_i)
        s_i_1, r_i_1, done, _ = env.step(a_i)
        ep.append(Step(s_i, a_i, r_i_1))
        s_i = s_i_1
        if done:
            break
    return ep


def mc_control(env, num_episodes, gamma, epsilon):
    q = defaultdict(lambda: defaultdict(float))
    policy = Policy(env, epsilon, q)
    returns = defaultdict(list)

    for i in range(num_episodes):
        ep = create_episode(env, policy)
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
        if i % 100 == 0:
            print("Ep %s: %s" % (i, check_win_ratio(env, policy, 10000)))
    return policy


def simulate_episode(env, policy):
    ep = []
    state = env.reset()
    for t in range(100):
        action = policy.choose_action(state)
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


if __name__ == "__main__":
    print("Training ....")
    policy = mc_control(env, 10000, 0.99, 0.05)
    print("eval: ")
    print(check_win_ratio(env, policy, 10000))


