import gym
import numpy as np
import tensorflow as tf


env = gym.make('CartPole-v1')
# model = tf.keras.models.load_model('model/20210611_165343/final_ep770/actor')
model = tf.keras.models.load_model('a2c_model/20210611_173809/final_ep2413/actor')
n_actions = 2

for i_episode in range(20):
    env.seed(i_episode)
    state = env.reset()
    print(i_episode, state)
    for t in range(500):
        env.render()
        prob = np.squeeze(model.predict(np.expand_dims(state, axis=0)))
        action = np.random.choice(np.arange(n_actions), p=prob)
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
