# ml/network.py
# =============================================================================
# DQN NEURAL NETWORK
# Simple MLP: state_size -> hidden -> hidden -> action_size
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
from config import CONFIG


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for cache replacement policy.

    Architecture: 3-layer MLP
        Input  : state vector (associativity × 4 features = 16 floats)
        Hidden : two layers of `hidden_size` neurons with ReLU
        Output : Q-value per action (one per cache way = 4 values)

    The network answers: "given this cache set state, how good is it
    to evict each of the N ways?"

    Weights are initialized with Xavier uniform (good for ReLU networks).
    """

    def __init__(self, cfg=CONFIG):
        super().__init__()
        state_size  = cfg['state_size']    # 16
        action_size = cfg['action_size']   # 4
        hidden_size = cfg['hidden_size']   # 64

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Xavier init — keeps gradients stable at start of training
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.device = torch.device('cpu')   # CPU only — no GPU needed for this scale

    def forward(self, x):
        """Raw forward pass. x: (batch, state_size) tensor."""
        return self.net(x)

    def get_q_values(self, state):
        """
        Get Q-values for a single state.

        Args:
            state: numpy array of shape (state_size,)

        Returns:
            numpy array of shape (action_size,) — one Q-value per cache way
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, state_size)
            q = self.net(t)                                              # (1, action_size)
        return q.squeeze(0).numpy()                                      # (action_size,)

    def get_action(self, state):
        """
        Greedy action: evict the way with the lowest Q-value.

        Low Q-value = evicting this block leads to the worst future outcome,
        i.e., it's the best block to evict (we want the cache to keep
        high-value blocks).

        Wait — actually high Q-value = evicting this block is GOOD
        (led to future hits). So we take argmax.

        Args:
            state: numpy array of shape (state_size,)

        Returns:
            int — index of way to evict (0 to action_size-1)
        """
        q_values = self.get_q_values(state)
        return int(np.argmax(q_values))

    def __repr__(self):
        total_params = sum(p.numel() for p in self.parameters())
        return (f"DQNNetwork(state={CONFIG['state_size']}, "
                f"hidden={CONFIG['hidden_size']}, "
                f"actions={CONFIG['action_size']}, "
                f"params={total_params})")
