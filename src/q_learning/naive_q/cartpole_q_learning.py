from q_learning.naive_q.tabular_q_learning import QTabularTrainer
from q_learning.naive_q.tabular_q_learning import EpsGreedyPolicy
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


MIN_MAX_POSITION = (-2.4, 2.4)
NUM_STATE_POSITION = 20
MIN_MAX_VELOCITY = (-3, 3)
NUM_STATE_VELOCITY = 20
MIN_MAX_ANGLE = (-0.418, 0.418)
NUM_STATE_ANGLE = 20
MIN_MAX_VELOCITY_AT_TIP = (-2, 2)
NUM_STATE_VELOCITY_AT_TIP = 20


def discrete_value(value, min_v, max_v, num_state):
    if value <= min_v:
        return 0
    elif value >= max_v:
        return num_state - 1
    state = int((value - min_v) / (max_v - min_v) * num_state)
    return state


class CatPoleQTabularTrainer(QTabularTrainer):
    def obs_to_state(self, obs):
        position, velocity, angle, velocity_at_tip = obs
        position = discrete_value(position, MIN_MAX_POSITION[0], MIN_MAX_POSITION[1], NUM_STATE_POSITION)
        velocity = discrete_value(velocity, MIN_MAX_VELOCITY[0], MIN_MAX_VELOCITY[1], NUM_STATE_VELOCITY)
        angle = discrete_value(angle, MIN_MAX_ANGLE[0], MIN_MAX_ANGLE[1], NUM_STATE_ANGLE)
        velocity_at_tip = discrete_value(velocity_at_tip, MIN_MAX_VELOCITY_AT_TIP[0],
                                         MIN_MAX_VELOCITY_AT_TIP[1], NUM_STATE_VELOCITY_AT_TIP)
        state_idx = [position, velocity, angle, velocity_at_tip]
        state_space = [NUM_STATE_POSITION, NUM_STATE_VELOCITY, NUM_STATE_ANGLE, NUM_STATE_VELOCITY_AT_TIP]
        return np.ravel_multi_index(state_idx, state_space)

    def train(self, n_episode):
        result = []
        grp_ep_size = []
        for i in range(n_episode):
            episode_size = 0
            obs = self.env.reset()
            s = self.obs_to_state(obs)
            while True:
                a = self.policy.select_action(s, self.Q)
                obs_1, r, done, _ = self.env.step(a)
                s1 = self.obs_to_state(obs_1)
                q_s1 = self.Q[s1]
                max_q_s1 = max(q_s1.values()) if len(q_s1) > 0 else 0
                self.Q[s][a] = self.Q[s][a] + self.alpha * (
                        r + self.gamma * max_q_s1 - self.Q[s][a]
                )
                s = s1
                episode_size = episode_size + 1
                if done:
                    result.append((i, episode_size))
                    if i % 100 == 0:
                        if i > 0:
                            print('Ep%s, %.1f' % (i, np.average(grp_ep_size)))
                        grp_ep_size = []
                    else:
                        grp_ep_size.append(episode_size)
                    break
        return result


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    actions = [x for x in range(env.action_space.n)]
    policy = EpsGreedyPolicy(eps=0.01, actions=actions)
    n_episode = 50000
    trainer = CatPoleQTabularTrainer(env=env, policy=policy, alpha=0.5, gamma=0.99)
    log = trainer.train(n_episode)

    print(trainer.Q)
    df_log = pd.DataFrame(log, columns=['ep', 'score'])
    df_log['grp'] = df_log.ep.apply(lambda x: int(x/100))
    df_log = df_log.groupby('grp').agg({'score': 'mean'}).reset_index().sort_values('grp')
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df_log.grp, df_log.score)
    plt.show()
    plt.close()

