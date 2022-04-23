import gym
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
from gym.envs.atari import AtariEnv
import math


def create_q_model(input_size, output_size, layers_sizes):
    inputs = Input(shape=input_size, dtype=tf.float32)
    x = Dense(units=layers_sizes[0], activation="relu")(inputs)
    for layer_size in layers_sizes[1:]:
        x = Dense(units=layer_size, activation="relu")(x)
    outputs = Dense(units=output_size, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_env():
    # env = AtariEnv(game='ms_pacman', obs_type='image')
    env = gym.make("MsPacman-v0")
    obs = env.reset()
    print(env.action_space)
    print(obs)
    print(obs.shape)
    while True:
        time.sleep(0.02)
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        if done:
            break
    env.close()


def obs_to_state(obs):
    return np.reshape(obs, -1) / 255.0


class QMemory(object):
    def __init__(self, max_memory_size):
        self.m_states = []
        self.m_actions = []
        self.m_rewards = []
        self.m_dones = []
        self.m_next_states = []
        self.max_memory_size = max_memory_size
        self.memory_size = 0

    @staticmethod
    def get_sample(memory, indices):
        return np.array(([memory[i] for i in indices]))

    def get_batch(self, batch_size):
        if self.memory_size > batch_size:
            indices = np.random.randint(low=0, high=self.memory_size, size=batch_size)
            states = QMemory.get_sample(self.m_states, indices)
            actions = QMemory.get_sample(self.m_actions, indices)
            rewards = QMemory.get_sample(self.m_rewards, indices)
            dones = QMemory.get_sample(self.m_dones, indices)
            next_states = QMemory.get_sample(self.m_next_states, indices)
            return states, actions, rewards, dones, next_states
        else:
            raise Exception("memory_size is not enough for batch_size")

    def insert(self, s, a, r, d, s1):
        if self.memory_size > self.max_memory_size:
            print("Exceeds max_memory_size. Reset memory: %s > %s " %
                  (self.memory_size, self.max_memory_size))
            self.reset_memory()
        self.m_states.append(s)
        self.m_actions.append(a)
        self.m_rewards.append(r)
        self.m_dones.append(d)
        self.m_next_states.append(s1)
        self.memory_size += 1

    def reset_memory(self):
        self.m_states = []
        self.m_actions = []
        self.m_rewards = []
        self.m_dones = []
        self.m_next_states = []
        self.memory_size = 0


class EpsPolicy(object):
    def get_action(self, state, q, n_acts, eps):
        output = q(tf.expand_dims(state, axis=0), training=False)
        best_action = tf.argmax(output, axis=1)
        prob = [
            1 - eps if a == best_action else eps / (n_acts - 1)
            for a in range(n_acts)
        ]
        return np.random.choice(range(n_acts), p=prob)


class BestActionPolicy(object):
    def get_action(self, state, q):
        output = q(tf.expand_dims(state, axis=0), training=False)
        best_action = tf.argmax(output, axis=1)
        return best_action


def debug_q_model(q):
    env = gym.make("MsPacman-v0")
    policy = BestActionPolicy()
    obs = env.reset()
    score = 0
    while True:
        state = obs_to_state(obs)
        env.render()
        action = policy.get_action(state, q)
        obs, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("- Total score: %s" % score)
            break
    env.close()



if __name__ == "__main__":
    # q_model setup
    input_size = 210 * 160 * 3
    n_acts = 9
    layers_sizes = [32, 64]
    q = create_q_model(input_size=input_size, output_size=n_acts, layers_sizes=layers_sizes)
    q_target = create_q_model(input_size=input_size, output_size=n_acts, layers_sizes=layers_sizes)
    q.summary()

    # training params
    n_epochs = 1000
    batch_size = 64
    # Q memory
    memory = QMemory(max_memory_size=100000)
    # learning params
    n_learning = 0
    learning_frame_interval = 64
    update_step_interval = 4
    n_frames = 0
    n_learning_steps = 0
    gamma = 0.99
    # exploration policy
    policy = EpsPolicy()
    base_eps = 0.5
    min_eps = 0.05
    max_eps = 0.5
    # Optimizer
    opt = Adam(learning_rate=0.1)

    # gym env
    env = AtariEnv(game='ms_pacman', obs_type='image')
    obs = env.reset()

    for i in range(n_epochs):
        obs = env.reset()
        cur_ep_n_frames = 0
        s = obs_to_state(obs)
        cur_eps = min(max_eps, max(min_eps, base_eps / math.log(i+2)))
        cur_score = 0
        while True:
            env.render()
            n_frames += 1
            cur_ep_n_frames += 1
            a = policy.get_action(s, q, n_acts, cur_eps)
            obs1, r, done, _ = env.step(a)
            cur_score += r
            s1 = obs_to_state(obs1)
            memory.insert(s, a, r, done, s1)
            s = s1
            if memory.memory_size > batch_size and n_frames % learning_frame_interval == 0:
                n_learning_steps += 1
                b_states, b_actions, b_rewards, b_dones, b_next_states = memory.get_batch(batch_size)
                target = b_rewards + (1 - b_dones) * gamma * \
                         tf.reduce_max(q_target(b_next_states, training=False), axis=1)
                b_acts_ohe = tf.one_hot(b_actions, n_acts)
                with tf.GradientTape() as tape:
                    pred = tf.reduce_max(q(b_states, training=False) * b_acts_ohe, axis=1)
                    loss = tf.square(target - pred)
                variables = q.trainable_variables
                grad = tape.gradient(loss, variables)
                opt.apply_gradients(zip(grad, variables))

            if n_learning_steps % update_step_interval == 0:
                q_target.set_weights(q.get_weights())

            if done:
                print("Ep %s: eps %.3f, score %s, frames %s, memory_size %s, n_learning_step %s" %
                      ((i+1), cur_eps, cur_score, cur_ep_n_frames, memory.memory_size, n_learning_steps))
                if (i+1) % 20 == 0:
                    print("- Check best policy at ep %s" % (i+1))
                    debug_q_model(q)
                break



