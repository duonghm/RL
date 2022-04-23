import tensorflow as tf
import numpy as np
import gym
import os
import time
from datetime import datetime
import tqdm


# tf.debugging.enable_check_numerics()
# tf.config.run_functions_eagerly(True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logdir = "logs/ac_scalars/" + cur_time
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


def to_discount_rewards(rewards: tf.Tensor,
                        gamma: float) -> tf.Tensor:
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


def step(action):
    next_state, reward, done, _ = env.step(action)
    return np.array(next_state, np.float32), np.float32(reward), np.bool(done)


@tf.function(experimental_relax_shapes=True)
def learn(actor: tf.keras.Model,
          critic: tf.keras.Model,
          actor_optimizer: tf.keras.optimizers.Optimizer,
          critic_optimizer: tf.keras.optimizers.Optimizer,
          gamma: float,
          init_state: tf.Tensor,
          ep: int) -> tf.Tensor:
    state = init_state
    base_state_shape = state.shape
    all_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    all_actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    all_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    idx = 0
    while tf.cast(True, tf.bool):
        action_prob = actor(tf.expand_dims(state, axis=0))
        action = tf.cast(tf.random.categorical(tf.math.log(action_prob), 1)[0,0], tf.int32)
        next_state, reward, done = tf.numpy_function(step, [action], [tf.float32, tf.float32, tf.bool])
        all_actions = all_actions.write(idx, action)
        all_states = all_states.write(idx, state)
        all_rewards = all_rewards.write(idx, reward)
        state = next_state
        state.set_shape(base_state_shape)
        if tf.cast(done, tf.bool):
            break
        idx = idx + 1
    all_states = all_states.stack()
    all_actions = all_actions.stack()
    all_rewards = all_rewards.stack()

    returns = to_discount_rewards(all_rewards, gamma)
    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(all_states))
        critic_loss = huber_loss(returns, values)
        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    tf.summary.scalar('critic_loss', data=tf.reduce_mean(critic_loss), step=ep)

    with tf.GradientTape() as actor_tape:
        action_probs = tf.reduce_max(
            actor(all_states) * tf.one_hot(all_actions, depth=actor.n_actions), axis=1
        )
        advantage = returns - values
        actor_loss = -(advantage * tf.math.log(action_probs))
        actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    tf.summary.scalar('actor_loss', data=tf.reduce_mean(actor_loss), step=ep)
    return tf.math.reduce_sum(all_rewards)


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
    rws = []
    base_model_folder = f'model/{cur_time}'
    with tqdm.trange(n_epochs) as t:
        for i in t:
            init_state = tf.cast(env.reset(), tf.float32)
            ep = tf.cast(i, tf.int64)
            ep_reward = int(learn(actor, critic, actor_opt, critc_opt, gamma, init_state, ep))
            rws.append(ep_reward)
            avg_rw = np.mean(rws)
            t.set_description(f'epoch {i}')
            t.set_postfix(ep_reward=ep_reward, avg_rw=avg_rw)
            tf.summary.scalar('reward', data=ep_reward, step=i)
            # if i % 100 == 0:
            #     actor.save(f'{base_model_folder}/ep{i}/actor')
            #     critic.save(f'{base_model_folder}/ep{i}/critic')
            if avg_rw >= 160 and i >= 100:
                break
    print(f'Finish at ep {i} - avg_rw {avg_rw}')
    actor.save(f'{base_model_folder}/final_ep{i}/actor')
    critic.save(f'{base_model_folder}/final_ep{i}/critic')
