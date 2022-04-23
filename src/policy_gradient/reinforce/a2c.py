import tensorflow as tf
import numpy as np
import gym
import os
from typing import List, Tuple
import tqdm
from datetime import datetime
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, LeakyReLU, Softmax


# tf.config.run_functions_eagerly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Actor(tf.keras.Model):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.mlp = tf.keras.Sequential([
            Dense(64), LeakyReLU(),
            Dense(128), LeakyReLU(),
            Dense(256), LeakyReLU(),
            Dense(n_actions), Softmax()
        ])
        self.n_actions = n_actions

    def call(self, inputs, training=None, mask=None):
        return self.mlp(inputs)


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.mlp = tf.keras.Sequential([
            Dense(256), BatchNormalization(), ReLU(),
            Dense(128), BatchNormalization(), ReLU(),
            Dense(64), BatchNormalization(), ReLU(),
            Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.mlp(inputs)


def step(action):
    next_state, reward, done, _ = env.step(action)
    return np.array(next_state, np.float32), np.float32(reward), np.float32(done)


def run_env(actor: tf.keras.Model,
            gamma: float,
            init_state: tf.Tensor
            ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    m_states = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    m_acts = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0)
    m_rewards = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    m_next_states = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    m_dones = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    m_gammas = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    m_size = 0
    state = init_state
    state_initial_shape = state.shape
    cur_gamma = 1.0
    while tf.cast(True, tf.bool):
        action_prob = actor(tf.expand_dims(state, axis=0), training=False)
        action = tf.cast(tf.random.categorical(tf.math.log(action_prob), 1)[0, 0], tf.int32)
        next_state, reward, done = tf.numpy_function(step, [action], [tf.float32, tf.float32, tf.float32])
        next_state.set_shape(state_initial_shape)
        # next_state.set_shape(init_shape)
        m_states = m_states.write(m_size, state)
        m_acts = m_acts.write(m_size, action)
        m_rewards = m_rewards.write(m_size, reward)
        m_next_states = m_next_states.write(m_size, next_state)
        m_dones = m_dones.write(m_size, done)
        m_gammas = m_gammas.write(m_size, cur_gamma)
        if tf.cast(done, tf.bool):
            break
        state = next_state
        cur_gamma = cur_gamma * gamma
        m_size += 1
    m_states = m_states.stack()
    m_acts = m_acts.stack()
    m_rewards = m_rewards.stack()
    m_next_states = m_next_states.stack()
    m_dones = m_dones.stack()
    m_gammas = m_gammas.stack()
    return m_states, m_acts, m_rewards, m_next_states, m_dones, m_gammas


# def sub_tf_arr(tf_arr: tf.TensorArray,
#                indicies: np.array) -> tf.Tensor:
#     return tf.gather(tf_arr.stack(), indicies)
#
#
# def get_data(batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
#     """return (state, action, reward, next_state)"""
#     if batch_size <= m_size:
#         indicies = np.random.randint(0, m_size, batch_size)
#         return sub_tf_arr(m_states, indicies), sub_tf_arr(m_actions, indicies), \
#                sub_tf_arr(m_rewards, indicies), sub_tf_arr(m_next_states, indicies)
#     else:
#         return m_states.stack(), m_actions.stack(), m_rewards.stack(), m_next_states.stack()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)


@tf.function
def train(actor: tf.keras.Model,
          critic: tf.keras.Model,
          actor_opt: tf.optimizers.Optimizer,
          critic_opt: tf.optimizers.Optimizer,
          gamma: float,
          init_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    states, acts, rewards, next_states, dones, gammas = run_env(actor, gamma, init_state)
    with tf.GradientTape(watch_accessed_variables=False) as cri_tape, \
            tf.GradientTape(watch_accessed_variables=False) as act_tape:
        cri_tape.watch(critic.trainable_variables)
        act_tape.watch(actor.trainable_variables)
        act_probs = tf.reduce_max(
            actor(states) * tf.one_hot(acts, depth=actor.n_actions), axis=1
        )
        values = critic(states, training=True)
        returns = rewards + gamma * (1.0 - dones) * critic(next_states, training=True)
        sigma = returns - values
        cri_loss = tf.reduce_mean(tf.math.square(sigma**2))
        act_loss = tf.reduce_mean(sigma * tf.math.log(act_probs) * gammas)
    cri_grads = cri_tape.gradient(cri_loss, critic.trainable_variables)
    critic_opt.apply_gradients(zip(cri_grads, critic.trainable_variables))
    act_grads = act_tape.gradient(act_loss, actor.trainable_variables)
    actor_opt.apply_gradients(zip(act_grads, actor.trainable_variables))
    return tf.reduce_mean(cri_loss), tf.reduce_mean(act_loss), tf.math.reduce_sum(rewards)


if __name__ == '__main__':
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_folder = f'a2c_model/{cur_time}'
    env = gym.make('CartPole-v0')
    n_epochs = 10000
    act_mdl = Actor(n_actions=2)
    cri_mdl = Critic()
    act_mdl.build((None, 4))
    cri_mdl.build((None, 4))
    # print(act_mdl.trainable_variables)
    act_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    cri_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    gamma = 0.99
    all_total_rewards = []
    with tqdm.trange(n_epochs) as t:
        for i in t:
            i_state = tf.cast(env.reset(), tf.float32)
            e_cri_loss, e_act_loss, e_reward = train(act_mdl, cri_mdl, act_opt, cri_opt, gamma, i_state)
            e_cri_loss, e_act_loss, e_reward = float(e_cri_loss), float(e_act_loss), float(e_reward)
            all_total_rewards.append(e_reward)
            avg_rewards = np.mean(all_total_rewards[-100:])
            t.set_description(f"Epoch {i}")
            t.set_postfix(reward_cur = e_reward, reward_avg=avg_rewards,
                          loss_cri=e_cri_loss, loss_act=e_act_loss)
            if avg_rewards >= 199 and i >= 100:
                break
    print(f'Finish at ep {i} - avg_rw {avg_rewards}')
    act_mdl.save(f'{base_model_folder}/final_ep{i}/actor')
    cri_mdl.save(f'{base_model_folder}/final_ep{i}/critic')

