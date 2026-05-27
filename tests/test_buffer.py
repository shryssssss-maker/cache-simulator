import unittest
import sys
import numpy as np
sys.path.insert(0, '..')
from ml.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def _get_cfg(self, capacity, min_size):
        return {'replay_buffer_size': capacity, 'min_buffer_size': min_size, 'batch_size': 32}

    def test_empty_not_ready(self):
        buf = ReplayBuffer(cfg=self._get_cfg(100, 10))
        self.assertFalse(buf.ready())

    def test_ready_after_min_size(self):
        buf = ReplayBuffer(cfg=self._get_cfg(100, 10))
        for _ in range(10):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        self.assertTrue(buf.ready())

    def test_capacity_not_exceeded(self):
        buf = ReplayBuffer(cfg=self._get_cfg(50, 10))
        for _ in range(100):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        self.assertEqual(len(buf), 50)

    def test_sample_correct_size(self):
        buf = ReplayBuffer(cfg=self._get_cfg(100, 10))
        for _ in range(50):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
        self.assertEqual(len(states), 8)
        self.assertEqual(len(actions), 8)
