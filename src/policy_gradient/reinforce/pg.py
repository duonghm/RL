import gym
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.initializers import GlorotUniform
import numpy as np

from datetime import datetime
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(4,)))
    model.add(Dense(units=20, name='dense1', activation='relu',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    model.add(Dense(units=10, name='dense2', activation='relu',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    model.add(Dense(units=2, name='output', activation='softmax',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    return model


class Reinforce(object):
    ep = 0
    def __init__(self, model, optimizer, gamma, n_acts):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_acts = n_acts

    def get_action_prob(self, states):
        return self.model(states)

    def do_action(self, state):
        prob = np.squeeze(self.model.predict(np.expand_dims(state, axis=0)))
        return np.random.choice(np.arange(self.n_acts), p=prob)

    def learn(self, states, actions, rewards):
        G = to_discount_rewards(rewards, self.gamma)
        with tf.GradientTape() as tape:
            action_probs = tf.reduce_max(
                self.model(states) * tf.one_hot(actions, depth=self.n_acts), axis=1
            )
            loss = tf.constant(-1.0) * G * tf.math.log(action_probs)
            vars = self.model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        tf.summary.scalar('loss', data=np.mean(loss), step=Reinforce.ep)
        Reinforce.ep += 1


def to_discount_rewards(rewards, gamma):
    g = []
    running_sum = 0
    for r in rewards[::-1]:
        running_sum = running_sum * gamma + r
        g.append(running_sum)
    g = np.array(g[::-1])
    g = (g - np.mean(g)) / np.std(g)
    return g


if __name__ == '__main__':
    np.random.seed(0)
    model = create_model()
    optimizer = Adam(learning_rate=0.001)
    model.summary()

    n_epochs = 1000
    env = gym.make('CartPole-v0')
    pg = Reinforce(model=model, optimizer=optimizer, gamma=0.99, n_acts=2)

    for i in range(n_epochs):
        all_states = []
        all_actions = []
        all_rewards = []
        state = env.reset()
        while True:
            action = pg.do_action(state)
            next_state, reward, done, _ = env.step(action)
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            state = next_state
            if done:
                break
        if i%10 == 0:
            print(i, len(all_states))
        pg.learn(np.array(all_states), np.array(all_actions), np.array(all_rewards))
        tf.summary.scalar('len', data=len(all_states), step=i)



