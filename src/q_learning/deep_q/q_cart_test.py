import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import gym


class BestActPolicy(object):
    def get_action(self, q, state):
        best_action = tf.argmax(
            q(tf.expand_dims(state, axis=0)), axis=1
        )[0].numpy()
        return best_action


def obs_to_state(obs):
    return np.array(obs)


if __name__ == "__main__":
    q = keras.models.load_model("model/model_ep_800.h5")
    q.summary()
    env = gym.make('CartPole-v1')
    policy = BestActPolicy()
    obs = env.reset()
    n_frame = 0
    while True:
        n_frame += 1
        env.render()
        s = obs_to_state(obs)
        a = policy.get_action(q, s)
        obs, r, done, _ = env.step(a)
        if done:
            print("finish after %s frames" % n_frame)
            break
    env.close()


