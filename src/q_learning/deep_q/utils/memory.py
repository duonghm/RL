import numpy as np


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
