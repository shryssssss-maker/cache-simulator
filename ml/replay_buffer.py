# ml/replay_buffer.py
# =============================================================================
# EXPERIENCE REPLAY BUFFER
# Stores (state, action, reward, next_state, done) tuples for DQN training
# =============================================================================

import numpy as np
import torch
import random
from collections import deque
from config import CONFIG


class ReplayBuffer:
    """
    Fixed-size circular replay buffer for DQN experience replay.

    Stores transitions: (state, action, reward, next_state, done)
    When full, oldest experiences are discarded (deque with maxlen).

    Training only starts once the buffer has at least `min_buffer_size`
    experiences — this ensures the first batches are diverse enough
    to be useful.

    Time complexity: O(1) push, O(batch_size) sample.
    Space complexity: O(replay_buffer_size × state_size).
    """

    def __init__(self, cfg=CONFIG):
        self.capacity       = cfg['replay_buffer_size']   # 10000
        self.batch_size     = cfg['batch_size']           # 32
        self.min_size       = cfg['min_buffer_size']      # 500

        # deque automatically evicts oldest entry when maxlen is reached
        self._buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done=False):
        """
        Add one transition to the buffer.

        Args:
            state:      numpy array (state_size,)
            action:     int — which way was evicted (0 to action_size-1)
            reward:     float — reward signal from environment
            next_state: numpy array (state_size,) — state after eviction
            done:       bool — episode ended? (always False for cache sim)
        """
        self._buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size=None):
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: number of transitions to sample (defaults to CONFIG)

        Returns:
            Tuple of 5 tensors:
                states      (batch, state_size)  float32
                actions     (batch,)             int64
                rewards     (batch,)             float32
                next_states (batch, state_size)  float32
                dones       (batch,)             float32 (0.0 or 1.0)
        """
        if batch_size is None:
            batch_size = int(self.batch_size)

        batch = random.sample(self._buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(np.array(actions),     dtype=torch.int64),
            torch.tensor(np.array(rewards),     dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones,  dtype=np.float32), dtype=torch.float32),
        )

    def ready(self):
        """
        Returns True when buffer has enough experiences to start training.
        Training before min_size leads to highly correlated batches.
        """
        return len(self._buffer) >= self.min_size

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return (f"ReplayBuffer(size={len(self._buffer)}/{self.capacity}, "
                f"ready={self.ready()})")
