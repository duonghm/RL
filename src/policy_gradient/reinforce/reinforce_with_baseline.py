import gym
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.initializers import GlorotUniform
import numpy as np
import time


from datetime import datetime
logdir = "logs/rb_scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


def create_policy_model():
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


def create_v_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(4,)))
    model.add(Dense(units=20, name='dense1', activation='relu',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    model.add(Dense(units=10, name='dense2', activation='relu',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    model.add(Dense(units=1, name='output', activation='linear',
                    kernel_initializer=GlorotUniform(seed=0),
                    bias_initializer=GlorotUniform(seed=0)))
    return model


class ReinfoceB(object):
    ep = 0

    def __init__(self, policy_model, v_model, policy_optimizer, v_optimizer, gamma, n_acts):
        self.policy_model = policy_model
        self.v_model = v_model
        self.policy_optimizer = policy_optimizer
        self.v_optimizer = v_optimizer
        self.gamma = gamma
        self.n_acts = n_acts

    def get_action_prob(self, states):
        return self.policy_model(states)

    def do_action(self, state):
        prob = np.squeeze(self.policy_model.predict(np.expand_dims(state, axis=0)))
        return np.random.choice(np.arange(self.n_acts), p=prob)

    def learn(self, states, actions, rewards):
        G = to_discount_rewards(rewards, self.gamma)
        with tf.GradientTape() as v_tape:
            # Update v_model
            sigma = G - self.v_model(states)
            sigma = (sigma - np.mean(sigma)) / np.std(sigma)
            v_loss = tf.constant(-1.0) * sigma
            v_vars = self.v_model.trainable_variables
            v_grads = v_tape.gradient(v_loss, v_vars)
            self.v_optimizer.apply_gradients(zip(v_grads, v_vars))
        tf.summary.scalar('v_loss', data=np.mean(v_loss), step=ReinfoceB.ep)

        with tf.GradientTape() as tape:
            action_probs = tf.reduce_max(
                self.policy_model(states) * tf.one_hot(actions, depth=self.n_acts), axis=1
            )
            loss = tf.constant(-1.0) * sigma * tf.math.log(action_probs)
            vars = self.policy_model.trainable_variables
            grads = tape.gradient(loss, vars)
            self.policy_optimizer.apply_gradients(zip(grads, vars))
        tf.summary.scalar('loss', data=np.mean(loss), step=ReinfoceB.ep)
        ReinfoceB.ep += 1


def to_discount_rewards(rewards, gamma):
    g = []
    running_sum = 0
    for r in rewards[::-1]:
        running_sum = running_sum * gamma + r
        g.append(running_sum)
    g = np.array(g[::-1])
    # g = (g - np.mean(g)) / np.std(g)
    return g


if __name__ == '__main__':
    np.random.seed(0)
    v_model = create_v_model()
    p_model = create_policy_model()
    v_optimizer = Adam(learning_rate=0.01)
    p_optimizer = Adam(learning_rate=0.01)
    v_model.summary()
    p_model.summary()

    n_epochs = 1000
    env = gym.make('CartPole-v0')
    pg = ReinfoceB(policy_model=p_model, v_model=v_model,
                   policy_optimizer=p_optimizer, v_optimizer=v_optimizer,
                   gamma=0.99, n_acts=2)
    start_time = time.time()
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
        if i % 10 == 0:
            print(i, len(all_states))
        pg.learn(np.array(all_states), np.array(all_actions), np.array(all_rewards))
        tf.summary.scalar('len', data=len(all_states), step=i)
        tf.summary.scalar('dur', data=round(time.time() - start_time, 0), step=i)
