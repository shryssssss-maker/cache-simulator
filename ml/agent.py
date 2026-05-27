# ml/agent.py
# =============================================================================
# DQN AGENT
# Combines network, replay buffer, epsilon-greedy, target network
# into a complete learning agent
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
from config import CONFIG
from ml.network import DQNNetwork
from ml.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Complete DQN agent for cache replacement.

    Components:
    1. Online network    → actively trained, makes decisions
    2. Target network    → frozen copy, provides stable training targets
    3. Replay buffer     → stores experiences, breaks correlation
    4. Epsilon-greedy    → balances exploration vs exploitation
    5. Safety guard      → prevents dirty evictions when clean available

    Training loop (called externally every N steps):
        sample batch from replay buffer
        compute target Q-values using target network
        compute current Q-values using online network
        compute loss (MSE between current and target)
        backpropagate and update online network
        periodically sync target network

    Eviction decision:
        with probability epsilon → random action (explore)
        with probability 1-epsilon → lowest Q-value action (exploit)
        safety guard → if DQN picks dirty block but clean available, override
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

        # Two networks: online (learning) and target (stable reference)
        self.online_network = DQNNetwork(cfg)
        self.target_network = copy.deepcopy(self.online_network)

        # Target network is NOT trained — only updated by copying online
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer for online network only
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=cfg['learning_rate']
        )

        self.replay_buffer  = ReplayBuffer(cfg)
        self.epsilon        = cfg['epsilon_start']

        self.step_count     = 0
        self.train_count    = 0
        self.total_loss     = 0.0
        self.loss_steps     = 0

        # For tracking safety guard usage
        self.last_was_override = False

        # For policy switcher compatibility
        self._last_state      = None
        self._last_action     = None
        self._last_evicted    = None

    # =========================================================================
    # EVICTION DECISION
    # =========================================================================

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select which block to evict from a full cache set.

        This is called by CacheSimulator._evict() — same interface
        as all other policies (FIFO, LRU, etc.)

        Decision process:
        1. Build state from cache set
        2. Epsilon-greedy: random or DQN
        3. Safety guard: override if DQN picks dirty when clean available
        4. Store state and action for later reward assignment

        Args:
            cache_set:    CacheSet object with blocks to choose from
            step:         current simulation step
            incoming_tag: unused (DQN doesn't need future info)

        Returns:
            CacheBlock to evict
        """
        self.step_count += 1
        self.last_was_override = False

        # Build normalized state vector
        state = cache_set.get_state(max_val=self.cfg['trace_length'])

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, len(cache_set.blocks) - 1)
        else:
            # Exploit: use DQN
            q_values = self.online_network.get_q_values(state)

            # Only consider valid block indices
            n_blocks = len(cache_set.blocks)
            valid_q  = q_values[:n_blocks]
            action   = int(np.argmax(valid_q))  # highest Q = evict this

            # Safety guard disabled for pure DQN testing
            # action = self._safety_guard(cache_set, action, q_values)

        # Store for reward assignment
        self._last_state   = state
        self._last_action  = action
        self._last_evicted = cache_set.blocks[action]

        return cache_set.blocks[action]

    def _safety_guard(self, cache_set, dqn_action, q_values):
        """
        Safety guard: prevent DQN from choosing dirty eviction when
        a clean block is available.

        During early training, DQN has random Q-values and might
        accidentally prefer dirty blocks. This guard corrects that
        until DQN learns the -0.2 dirty penalty itself.

        When safety guard fires less often over training → DQN learned ✅
        When safety guard always fires → DQN never learned ❌

        Args:
            cache_set:  CacheSet
            dqn_action: action DQN wants to take
            q_values:   all Q-values from DQN

        Returns:
            corrected action index
        """
        chosen_block = cache_set.blocks[dqn_action]

        if not chosen_block.dirty:
            # DQN already picked clean — no override needed
            return dqn_action

        # DQN picked dirty — find clean alternatives
        clean_indices = [i for i, b in enumerate(cache_set.blocks)
                         if not b.dirty]

        if not clean_indices:
            # All blocks are dirty — no override possible
            return dqn_action

        # Override: pick clean block with lowest Q-value
        self.last_was_override = True
        clean_q_values = [(i, q_values[i]) for i in clean_indices]
        best_clean     = min(clean_q_values, key=lambda x: x[1])
        return best_clean[0]

    # =========================================================================
    # REWARD AND LEARNING
    # =========================================================================

    def record_reward(self, next_cache_state, access_result):
        """
        Calculate reward and store experience in replay buffer.

        Call this AFTER the eviction and the NEXT access have been processed.
        This gives us the consequence of the eviction decision.

        Reward structure:
            +1.0  → next access was a HIT  (good eviction)
            -1.0  → next access was a MISS (bad eviction)
            -0.2  → evicted block was dirty (writeback penalty)
            +0-0.3 → recency bonus (evicting old blocks is probably good)

        Args:
            next_cache_state: CacheSet state after loading new block
            access_result:    dict from CacheSimulator.access()
        """
        if self._last_state is None:
            return  # no eviction happened yet

        # Calculate reward
        reward = self._calculate_reward(access_result)

        # Build next state
        next_state = np.array(next_cache_state, dtype=np.float32)

        # Store experience
        self.replay_buffer.push(
            self._last_state,
            self._last_action,
            reward,
            next_state,
        )

        # Reset tracking
        self._last_state   = None
        self._last_action  = None

    def _calculate_reward(self, next_result):
        """
        Calculate reward for the previous eviction decision.

        Args:
            next_result: result dict from the access AFTER eviction

        Returns:
            float reward
        """
        reward = 0.0

        # Primary signal: was the next access a hit or miss?
        if next_result.get('hit', False):
            reward += self.cfg['hit_reward']      # +1.0
        else:
            reward += self.cfg['miss_penalty']    # -1.0

        # Dirty penalty: was the evicted block dirty?
        if self._last_evicted and self._last_evicted.dirty:
            reward += self.cfg['dirty_penalty']   # -0.2

        # Recency bonus: reward evicting blocks that haven't been used recently
        if self._last_evicted:
            max_recency  = self.cfg['trace_length']
            recency_norm = min(self._last_evicted.recency / max_recency, 1.0)
            bonus        = recency_norm * self.cfg['recency_bonus_max']
            reward      += bonus                  # 0 to +0.3

        return reward

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train_step(self):
        """
        Perform one training step: sample batch, compute loss, update network.

        Call this every N simulation steps (N = train_frequency from config).
        Only trains if buffer has enough experiences.

        Training algorithm (DQN with target network):
        1. Sample random batch of 32 experiences
        2. For each experience:
           target_q = reward + gamma * min(target_network(next_state))
        3. Compute MSE loss between online_network(state)[action] and target_q
        4. Backpropagate, update online network weights

        Returns:
            float loss value, or None if buffer not ready
        """
        if not self.replay_buffer.ready():
            return None

        # Sample random batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample()

        # Convert to tensors
        states_t      = states
        actions_t     = actions
        rewards_t     = rewards
        next_states_t = next_states
        dones_t       = dones

        # Current Q-values from online network
        # Shape: (batch_size, action_size)
        current_q_all  = self.online_network(states_t)

        # Q-values for the actions that were actually taken
        # Shape: (batch_size,)
        current_q = current_q_all.gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values from TARGET network (frozen)
        with torch.no_grad():
            next_q_all = self.target_network(next_states_t)
            # We want to maximize reward → use max not min
            next_q_max = next_q_all.max(dim=1)[0]

            # Bellman equation:
            # target = reward + gamma * max_next_q * (1 - done)
            target_q = rewards_t + \
                       self.cfg['gamma'] * next_q_max * (1 - dones_t)

        # Compute MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=1.0
        )

        self.optimizer.step()

        # Track loss
        loss_val        = loss.item()
        self.total_loss += loss_val
        self.loss_steps += 1
        self.train_count += 1

        # Sync target network periodically
        if self.train_count % self.cfg['target_sync_steps'] == 0:
            self._sync_target_network()

        return loss_val

    def _sync_target_network(self):
        """
        Copy online network weights to target network.
        Called every target_sync_steps training steps.
        """
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def decay_epsilon(self):
        """
        Decay epsilon by one step.
        Call this after each simulation step.
        """
        self.epsilon = max(
            self.cfg['epsilon_end'],
            self.epsilon * self.cfg['epsilon_decay']
        )

    def get_avg_loss(self):
        """Get average loss since last call. Resets counter."""
        if self.loss_steps == 0:
            return 0.0
        avg = self.total_loss / self.loss_steps
        self.total_loss = 0.0
        self.loss_steps = 0
        return avg

    # =========================================================================
    # CONFIDENCE CHECK (for policy switcher)
    # =========================================================================

    def get_confidence(self, state):
        """
        Measure agent confidence: gap between best and second-best Q-value.
        High gap → agent is sure about its decision.
        Low gap  → agent is uncertain, better to use LRU.

        Args:
            state: numpy array of state features

        Returns:
            float confidence score
        """
        q_values = self.online_network.get_q_values(state)
        sorted_q = sorted(q_values)     # ascending
        # Confidence = gap between 2nd lowest and lowest
        # (difference between best eviction and second-best eviction)
        if len(sorted_q) >= 2:
            return sorted_q[1] - sorted_q[0]
        return 0.0

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path=None):
        """Save trained model to disk."""
        if path is None:
            path = self.cfg['model_path']
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'online_state_dict' : self.online_network.state_dict(),
            'target_state_dict' : self.target_network.state_dict(),
            'optimizer_state'   : self.optimizer.state_dict(),
            'epsilon'           : self.epsilon,
            'step_count'        : self.step_count,
            'train_count'       : self.train_count,
        }, path)
        print(f"Model saved -> {path}")

    def load(self, path=None):
        """Load trained model from disk."""
        if path is None:
            path = self.cfg['model_path']
        checkpoint = torch.load(path, map_location='cpu')
        self.online_network.load_state_dict(
            checkpoint['online_state_dict']
        )
        self.target_network.load_state_dict(
            checkpoint['target_state_dict']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon    = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.train_count = checkpoint['train_count']
        self.online_network.eval()
        self.target_network.eval()
        print(f"Model loaded <- {path}")

    def reset(self):
        """Reset agent for new simulation (keeps learned weights)."""
        self._last_state   = None
        self._last_action  = None
        self._last_evicted = None
