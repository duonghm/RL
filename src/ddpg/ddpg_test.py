import gym
import tensorflow as tf
import numpy as np


def policy(env, act, state):
    upper_bond = env.action_space.high[0]
    lower_bond = env.action_space.low[0]
    act_sample = np.squeeze(act(state))
    act_sample = np.clip(act_sample, lower_bond, upper_bond)
    return [act_sample]


if __name__ == '__main__':
    actor = tf.keras.models.load_model('ddpg_models/20210913_170332/ep49/actor')
    actor.summary()
    env = gym.make('Pendulum-v0')
    state = env.reset()
    while True:
        env.render()
        state = tf.expand_dims(state, 0)
        action = policy(env, actor, state)
        # action = np.random.uniform(-0.01, 0.01, 1)
        next_state, reward, done, info = env.step(action)

        if done:
            break
        state = next_state
    env.close()
