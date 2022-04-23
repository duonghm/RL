import tensorflow as tf
import numpy as np
import gym
import os
import time
from datetime import datetime


tf.debugging.enable_check_numerics()
tf.config.run_functions_eagerly(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logdir = "logs/ac_scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


class Actor(tf.keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.dense1 = tf.keras.layers.Dense(units=20,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=n_actions,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                                  bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                                  activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

    def do_action(self, state):
        prob = np.squeeze(self.predict(np.expand_dims(state, axis=0)))
        return np.random.choice(np.arange(self.n_actions), p=prob)


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=20,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=10,
                                            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                            activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1,
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                                  bias_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                                                  activation='linear')

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


def to_discount_rewards(rewards, gamma):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]
    returns = ((returns - tf.math.reduce_mean(returns)) /
                   (tf.math.reduce_std(returns) + 1e-7))
    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


@tf.function
def learn(actor, critic, actor_optimizer, critic_optimizer, states, rewards, actions, gamma, ep):
    returns = to_discount_rewards(rewards, gamma)
    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(states))
        critic_loss = huber_loss(returns, values)
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    tf.summary.scalar('loss_critic', data=tf.reduce_mean(critic_loss), step=ep)

    with tf.GradientTape() as actor_tape:
        action_probs = tf.reduce_max(
            actor(states) * tf.one_hot(actions, depth=actor.n_actions), axis=1
        )
        advantage = returns - values
        actor_loss = -(advantage * tf.math.log(action_probs))
        actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    tf.summary.scalar('loss_actor', data=tf.reduce_mean(actor_loss), step=ep)


# @tf.function
# def learn(actor, critic, actor_optimizer, critic_optimizer, states, rewards, actions, gamma, n_actions, ep):
#     returns = to_discount_rewards(rewards, gamma)
#     with tf.GradientTape() as critic_tape:
#         values = critic(states)
#         sigma = returns - values
#         sigma = (sigma - tf.reduce_mean(sigma)) / tf.math.reduce_std(sigma)
#         critic_loss = - sigma
#         critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
#     critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
#     tf.summary.scalar('loss_critic', data=tf.reduce_mean(critic_loss), step=ep)
#
#     with tf.GradientTape() as actor_tape:
#         action_probs = tf.reduce_max(
#             actor(states) * tf.one_hot(actions, depth=n_actions), axis=1
#         )
#         actor_loss = - sigma * tf.math.log(action_probs)
#         actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
#     actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
#     tf.summary.scalar('loss_actor', data=tf.reduce_mean(actor_loss), step=ep)


if __name__ == '__main__':
    np.random.seed(0)
    env = gym.make('CartPole-v0')
    n_epochs = 1000
    gamma = 0.99
    n_actions = 2
    actor = Actor(n_actions)
    critic = Critic()
    actor_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    critc_opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    start_time = time.time()
    for i in range(n_epochs):
        all_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        all_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        all_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        state = env.reset()
        idx = 0
        while True:
            action_prob = np.squeeze(actor.predict(np.expand_dims(state, axis=0)))
            action = np.random.choice(np.arange(n_actions), p=action_prob)
            next_state, reward, done, _ = env.step(action)
            all_actions = all_actions.write(idx, action)
            all_states = all_states.write(idx, state)
            all_rewards = all_rewards.write(idx, reward)
            state = next_state
            if done:
                break
            idx = idx + 1
        all_states = all_states.stack()
        all_actions = all_actions.stack()
        all_rewards = all_rewards.stack()
        if i % 10 == 0:
            print(i, len(all_states), round(time.time() - start_time, 0))
        learn(actor, critic, actor_opt, critc_opt, all_states, all_rewards, all_actions, gamma, i)
        tf.summary.scalar('len', data=len(all_states), step=i)
        tf.summary.scalar('dur', data=round(time.time() - start_time, 0), step=i)
