import tensorflow as tf
from tensorflow.keras.layers import Dense, ReLU
import gym
import numpy as np
import os
from datetime import datetime


# tf.config.run_functions_eagerly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.mdl = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1, activation='tanh',
                  kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        ])

    def call(self, inputs, training=None, mask=None):
        return self.mdl(inputs)


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.hidden = tf.keras.Sequential([
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1)
        ])
        self.state_hidden = tf.keras.Sequential([
            Dense(16, activation="relu"),
            Dense(32, activation="relu")
        ])
        self.action_hidden = tf.keras.Sequential([
            Dense(32, activation="relu")
        ])

    def call(self, inputs, training=None, mask=None):
        state_out = self.state_hidden(inputs[0])
        action_out = self.action_hidden(inputs[1])
        input = tf.keras.layers.Concatenate()([state_out, action_out])
        return self.hidden(input)


mse = tf.keras.losses.MeanSquaredError()


@tf.function
def train_critic(cri: tf.keras.Model,
                 act_target: tf.keras.Model,
                 cri_target: tf.keras.Model,
                 cri_opt: tf.keras.optimizers.Optimizer,
                 states, actions, rewards, next_states,
                 gamma: float
                 ):
    with tf.GradientTape() as tape:
        target_actions = act_target(next_states, training=True)
        y = rewards + gamma * cri_target([next_states, target_actions], training=True)
        cri_value = cri([states, actions], training=True)
        cri_loss = mse(y, cri_value)
    cri_grad = tape.gradient(cri_loss, cri.trainable_variables)
    cri_opt.apply_gradients(zip(cri_grad, cri.trainable_variables))


@tf.function
def train_actor(act: tf.keras.Model,
                cri: tf.keras.Model,
                act_opt: tf.keras.optimizers.Optimizer,
                states
                ):
    with tf.GradientTape() as tape:
        actions = act(states, training=True)
        cri_value = cri([states, actions], training=True)
        act_loss = - tf.reduce_mean(cri_value)
    act_grad = tape.gradient(act_loss, act.trainable_variables)
    act_opt.apply_gradients(zip(act_grad, act.trainable_variables))


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class Buffer(object):
    def __init__(self, max_size, state_dim, action_dim):
        self.cur_size = 0
        self.max_size = max_size
        self.buffer_states = np.zeros((max_size, state_dim))
        self.buffer_actions = np.zeros((max_size, action_dim))
        self.buffer_rewards = np.zeros((max_size, 1))
        self.buffer_next_states = np.zeros((max_size, state_dim))

    def insert(self, state, action, reward, next_state):
        idx = self.cur_size % self.max_size
        self.buffer_states[idx] = state
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_next_states[idx] = next_state
        self.cur_size += 1

    def sample(self, batch_size):
        idx_range = min(self.cur_size, self.max_size)
        indexes = np.random.choice(idx_range, batch_size)
        states = tf.convert_to_tensor(self.buffer_states[indexes])
        actions = tf.convert_to_tensor(self.buffer_actions[indexes])
        rewards = tf.cast(tf.convert_to_tensor(self.buffer_rewards[indexes]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.buffer_next_states[indexes])
        return states, actions, rewards, next_states


def policy(env, act, state):
    upper_bond = env.action_space.high[0]
    lower_bond = env.action_space.low[0]
    act_sample = np.squeeze(act(state))
    act_sample = np.clip(act_sample, lower_bond, upper_bond)
    return [act_sample]


if __name__ == '__main__':
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_folder = f'ddpg_models/{cur_time}'

    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(state_dim, action_dim)

    actor = Actor()
    actor.build((None, state_dim))
    actor.summary()
    critic = Critic()
    critic.build([(None, state_dim), (None, action_dim)])
    critic.summary()

    actor_target = Actor()
    actor_target.build((None, state_dim))
    critic_target = Critic()
    critic_target.build([(None, state_dim), (None, action_dim)])
    actor_target.set_weights(actor.get_weights())
    critic_target.set_weights(critic.get_weights())

    actor_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    buffer = Buffer(max_size=10000, state_dim=state_dim, action_dim=action_dim)

    n_epochs = 100
    batch_size = 64
    gamma = 0.99
    tau = 0.005

    rewards_collection = []
    for ep in range(n_epochs):
        state = env.reset()
        ep_reward = 0
        while True:
            state = tf.expand_dims(state, 0)
            action = policy(env, actor, state)  # TODO: Add exploration noise
            next_state, reward, done, info = env.step(action)
            # print(ep, action, reward, info)
            buffer.insert(state, action, reward, next_state)

            ep_reward += reward
            # Learn
            s, a, r, s1 = buffer.sample(batch_size)
            train_critic(critic, actor_target, critic_target, critic_opt, s, a, r, s1, gamma)
            train_actor(actor, critic, actor_opt, s)
            # Update target
            update_target(critic_target.variables, critic.variables, tau)
            update_target(actor_target.variables, actor.variables, tau)
            # End ep
            if done:
                break
            state = next_state
        print(f"ep {ep}: {ep_reward}")
        if len(rewards_collection) == 0 or ep_reward > max(rewards_collection):
            actor.save(f"{base_model_folder}/ep{ep}/actor")
            critic.save(f"{base_model_folder}/ep{ep}/critic")
        rewards_collection.append(ep_reward)
