import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import gym
import time


# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class EpsPolicy(object):
    def get_action(self, q, state, n_acts, eps):
        output = q(tf.expand_dims(state, axis=0), training=False)
        best_action = tf.argmax(output, axis=1)[0].numpy()
        prob = [
            1 - eps if a == best_action else eps / (n_acts - 1)
            for a in range(n_acts)
        ]
        action = np.random.choice(range(n_acts), p=prob)
        # print("output: %s - prob: %s - action: %s" % (output, prob, action))
        return action


class BestActPolicy(object):
    def get_action(self, q, state):
        best_action = tf.argmax(
            q(tf.expand_dims(state, axis=0), training=False), axis=1
        )[0].numpy()
        return best_action


def obs_to_state(obs):
    return np.array(obs)


def create_q_model(state_dim, n_acts, layers_sizes):
    inputs = Input(shape=(state_dim,), dtype=tf.float32)
    x = Dense(units=layers_sizes[0], activation="relu")(inputs)
    for layer_size in layers_sizes[1:]:
        x = Dense(units=layer_size, activation="relu")(x)
    outputs = Dense(units=n_acts, activation="linear")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def debug(q):
    env = gym.make('CartPole-v1')
    p = BestActPolicy()
    obs = env.reset()
    length = 0
    while True:
        length = length + 1
        env.render()
        a = p.get_action(q, obs_to_state(obs))
        obs1, r, done, _ = env.step(a)
        obs = obs1
        if done:
            print(" - Frame length: %s" % length)
            break
    env.close()
    return length


if __name__ == "__main__":
    # Create q model
    state_dim = 4
    n_acts = 2
    hidden_layers = [32, 32]
    batch_size = 32
    q = create_q_model(state_dim, n_acts, hidden_layers)
    q_target = create_q_model(state_dim, n_acts, hidden_layers)
    optimizer = Adam(learning_rate=0.01)
    # Create env
    env = gym.make('CartPole-v1')
    n_episodes = 500
    gamma = 0.99
    alpha = 0.5
    base_eps = 0.5
    min_eps = 0.05
    m_states = []
    m_actions = []
    m_rewards = []
    m_dones = []
    m_next_states = []
    m_len = 0
    max_memory_size = 100000
    n_learn_step = 0
    frame_step = 4
    target_update_step = 4

    policy = EpsPolicy()
    for i in range(n_episodes):
        obs = env.reset()
        s = obs_to_state(obs)
        cur_frame_len = 0
        while True:
            cur_frame_len += 1
            a = policy.get_action(q=q, state=s, n_acts=n_acts, eps=max(min_eps, base_eps/(i+1)))
            next_obs, r, done, _ = env.step(a)
            next_state = obs_to_state(next_obs)
            m_states.append(s)
            m_actions.append(a)
            m_rewards.append(r)
            m_dones.append(done)
            m_next_states.append(next_state)
            m_len = m_len + 1

            if m_len > batch_size and m_len % frame_step == 0:
                n_learn_step = n_learn_step + 1
                indices = np.random.randint(0, m_len, batch_size)
                state_batch = np.array([m_states[i] for i in indices])
                action_batch = np.array([m_actions[i] for i in indices])
                next_state_batch = np.array([m_next_states[i] for i in indices])
                reward_batch = np.array([m_rewards[i] for i in indices])
                done_batch = np.array([m_dones[i] for i in indices])

                target = reward_batch + (1 - done_batch) * gamma * tf.reduce_max(q_target(next_state_batch), axis=1)
                with tf.GradientTape() as tape:
                    acts_ohe = tf.one_hot(action_batch, n_acts)
                    pred = tf.reduce_max(q(state_batch) * acts_ohe, axis=1)
                    loss = tf.square(pred-target)
                    # loss = tf.reduce_mean(tf.square(pred - target))
                    # print("ep %s - loss %s: %s" % (i, n_learn_step, tf.reduce_mean(loss).numpy()))
                vars = q.trainable_variables
                grads = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars))

                if n_learn_step % target_update_step == 0:
                    # print("update q_target at n_learn_step: %s" % n_learn_step)
                    q_target.set_weights(q.get_weights())

            s = next_state
            if done:
                print("- Epoch %s: %s" % (i, cur_frame_len))
                # time.sleep(0.1)
                break
        if m_len > max_memory_size:
            print("m_len = %s excess maximum memory. Reset memory !" % m_len)
            m_states = []
            m_actions = []
            m_rewards = []
            m_dones = []
            m_next_states = []
            m_len = 0

        if (i+1) % 20 == 0:
            print("Epoch %d" % (i+1))
            print(q.get_weights())
            d_length = debug(q)
            q.save("model/model_ep_%s.h5" % (i+1))
            time.sleep(1)
            if d_length == 500:
                print("Reach goal. Stop training !")
                break


