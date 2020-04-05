from gym.envs import toy_text
from collections import defaultdict
from collections import namedtuple
import numpy as np
import monte_carlo.plotting as plotting
env = toy_text.BlackjackEnv()
env.seed(0)


def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1


Step = namedtuple('Step', ('state', 'action', 'reward'))


def generate_episode(policy, env):
    ep = []
    state = env.reset()
    for t in range(100):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        ep.append(Step(state, action, reward))  # (S_i, A_i, R_(i+1))
        state = next_state
        if done:
            break
    return ep


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    # returns_sum = defaultdict(float)
    # returns_count = defaultdict(float)
    returns = defaultdict(list)

    # The final value function
    V = defaultdict(float)

    for i in range(num_episodes):
        ep = generate_episode(policy, env)
        print('=' * 50)
        print('Epoch %d' % i)
        print(ep)
        g = 0
        visited_states = []
        for step in ep[::-1]:
            # print(visited_states)
            g = discount_factor * g + step.reward
            if step.state not in visited_states:
                returns[step.state].append(g)
                V[step.state] = np.mean(returns[step.state])
            visited_states.append(step.state)
    # for state in returns.keys():
    #     V[state] = np.mean(returns[state])
    return V, returns


def mc_prediction2(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        # if i_episode % 1000 == 0:
        #     print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        #     sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V


V_10k, returns_10k = mc_prediction(sample_policy, env, num_episodes=10000, discount_factor=0.9)

print('=== Value_function ===')
for state, v in V_10k.items():
    print(state, v)
print('==== returns ===')
for state, v in returns_10k.items():
    print(state, v, np.average(v))


plotting.plot_value_function(V_10k, title="10,000 Steps")

# V_10k_2 = mc_prediction2(sample_policy, env, num_episodes=100)
# plotting.plot_value_function(V_10k_2, title="10,000 Steps")

# V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
# plotting.plot_value_function(V_500k, title="500,000 Steps")