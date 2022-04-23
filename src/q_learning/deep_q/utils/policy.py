import tensorflow as tf
import numpy as np


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
