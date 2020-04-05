import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import dill


MIN_MAX_POSITION = (-2.4, 2.4)
NUM_STATE_POSITION = 20
MIN_MAX_VELOCITY = (-3, 3)
NUM_STATE_VELOCITY = 30
MIN_MAX_ANGLE = (-0.418, 0.418)
NUM_STATE_ANGLE = 30
MIN_MAX_VELOCITY_AT_TIP = (-2, 2)
NUM_STATE_VELOCITY_AT_TIP = 30


class CartPoleStateConverter(object):
    def __init__(self):
        pass

    def discrete_value(self, value, min_v, max_v, num_state):
        if value <= min_v:
            return 0
        elif value >= max_v:
            return num_state - 1
        state = int((value - min_v) / (max_v - min_v) * num_state)
        return state

    def to_state(self, obs):
        position, velocity, angle, velocity_at_tip = obs
        position = self.discrete_value(position, MIN_MAX_POSITION[0], MIN_MAX_POSITION[1], NUM_STATE_POSITION)
        velocity = self.discrete_value(velocity, MIN_MAX_VELOCITY[0], MIN_MAX_VELOCITY[1], NUM_STATE_VELOCITY)
        angle = self.discrete_value(angle, MIN_MAX_ANGLE[0], MIN_MAX_ANGLE[1], NUM_STATE_ANGLE)
        velocity_at_tip = self.discrete_value(velocity_at_tip, MIN_MAX_VELOCITY_AT_TIP[0],
                                         MIN_MAX_VELOCITY_AT_TIP[1], NUM_STATE_VELOCITY_AT_TIP)
        state_idx = [position, velocity, angle, velocity_at_tip]
        state_space = [NUM_STATE_POSITION, NUM_STATE_VELOCITY, NUM_STATE_ANGLE, NUM_STATE_VELOCITY_AT_TIP]
        return np.ravel_multi_index(state_idx, state_space)


class SarsaControl(object):
    def __init__(self,
                 env: gym.Env,
                 eps: float,
                 q=None,
                 is_debug=True
                 ):
        self.env = env
        self.eps = eps
        self.q = defaultdict(lambda: defaultdict(float)) if q is None else q
        self.state_converter = CartPoleStateConverter()
        self.is_debug = is_debug

    def policy(self, state):
        if len(self.q[state]) > 0:
            best_action = max(self.q[state], key=self.q[state].get)
            action_size = self.env.action_space.n
            probs = {
                action: 1 - self.eps + self.eps / action_size if action == best_action else self.eps / action_size
                for action in range(action_size)
            }
            action = np.random.choice(list(probs.keys()), p=list(probs.values()))
        else:
            action = self.env.action_space.sample()
        return action

    def obs_to_state(self, obs):
        state = self.state_converter.to_state(obs)
        return state

    def train(self, num_ep, alpha, gamma):
        result = []
        for i in range(num_ep):
            is_display = self.is_debug and (i+1) % 1000 == 0
            n_frame = 1
            obs = self.env.reset()
            state = self.obs_to_state(obs)
            action = self.policy(state)
            while True:
                if is_display:
                    self.env.render()
                n_frame = n_frame + 1
                next_obs, reward, done, _ = self.env.step(action)
                q_sa = self.q[state][action]
                if not done:
                    next_state = self.obs_to_state(next_obs)
                    next_action = self.policy(next_state)
                    next_q_sa = self.q[next_state][next_action]
                    q_sa += alpha * (reward + gamma * next_q_sa - q_sa)
                    self.q[state][action] = q_sa
                    state = next_state
                    action = next_action
                else:
                    q_sa += alpha * (reward - q_sa)
                    self.q[state][action] = q_sa
                    break


            if (i+1) % 1000 == 0:
                print(i+1, n_frame)
            result.append((i+1, n_frame))
        return result


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_ep = 50000
    sarsa_control = SarsaControl(env, eps=0.01, is_debug=False)
    training_result = sarsa_control.train(n_ep, alpha=0.1, gamma=0.99)
    env.close()
    dill.dump(sarsa_control.q, open('%s.q' % n_ep, 'wb'))
    df = pd.DataFrame(training_result, columns=['ep', 'n_frame'])
    df['grp'] = df['ep'].apply(lambda x: int((x - 1) / 100))
    df_result = df.groupby('grp').agg({'n_frame': 'mean'}).reset_index()
    plt.figure(figsize=(15,5))
    plt.plot(df_result.grp, df_result.n_frame, '--.')
    plt.show()
    plt.close()
