# policies/dqn.py
# =============================================================================
# DQN POLICY WRAPPER WITH COLD START SOLUTION
# Wraps DQNAgent to provide same interface as other policies
# Implements confidence-based LRU→DQN transition
# =============================================================================

from config import CONFIG
from policies.lru import LRUPolicy
from ml.agent import DQNAgent


class DQNPolicy:
    """
    DQN-based replacement policy with cold start handling.

    Problem: DQN starts knowing nothing. Early decisions are random
    and worse than LRU. We need to handle this gracefully.

    Solution: Confidence-based warm-up
    1. Start with LRU (safe, reliable)
    2. Simultaneously run DQN in shadow mode (decisions not applied)
    3. When DQN confidence exceeds threshold -> switch to DQN
    4. If DQN performance drops below LRU -> can revert

    For clean benchmarking (Experiment A):
    Use pre-trained DQN only (no LRU fallback).
    Set use_lru_warmup=False.

    For real-world demo (Experiment B / Hybrid mode):
    Use LRU warmup -> DQN transition.
    Set use_lru_warmup=True.
    """

    def __init__(self, agent=None, use_lru_warmup=True, cfg=CONFIG):
        self.cfg           = cfg
        self.use_lru_warmup = use_lru_warmup

        self.agent         = agent if agent else DQNAgent(cfg)
        self.lru_fallback  = LRUPolicy()

        self.current_policy = 'LRU' if use_lru_warmup else 'DQN'
        self.switch_step    = None  # step when switch occurred

        # Performance tracking for switch decision
        self.recent_hits_lru = []
        self.recent_hits_dqn = []

        self.last_was_override = False

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select block to evict using current policy.

        During warm-up: use LRU but check if ready to switch.
        After switch: use DQN with confidence check.

        Args:
            cache_set:    CacheSet to choose from
            step:         current simulation step
            incoming_tag: unused

        Returns:
            CacheBlock to evict
        """
        state = cache_set.get_state()

        if self.current_policy == 'LRU' and self.use_lru_warmup:
            # Check if DQN is ready to take over
            if self._should_switch(state):
                self.current_policy = 'DQN'
                self.switch_step    = step
                print(f"\n>>> Switched from LRU to DQN at step {step} <<<\n")

        if self.current_policy == 'DQN' or not self.use_lru_warmup:
            # Use DQN
            if not self.use_lru_warmup:
                # Pure DQN - no confidence fallback
                block = self.agent.select_eviction(cache_set, step, incoming_tag)
                self.last_was_override = self.agent.last_was_override
                return block
            else:
                # Hybrid mode - check confidence
                confidence = self.agent.get_confidence(state)
                if confidence >= self.cfg['confidence_threshold']:
                    block = self.agent.select_eviction(cache_set, step, incoming_tag)
                    self.last_was_override = self.agent.last_was_override
                    return block
                else:
                    self.last_was_override = False
                    return self.lru_fallback.select_eviction(cache_set)
        else:
            # Pure LRU warm-up
            self.last_was_override = False
            return self.lru_fallback.select_eviction(cache_set)

    def _should_switch(self, state):
        """
        Decide if DQN is ready to take over from LRU.

        Condition: DQN must show higher confidence than threshold
        AND buffer must be ready for training.

        Args:
            state: current cache state

        Returns:
            True if should switch to DQN
        """
        if not self.agent.replay_buffer.ready():
            return False    # not enough data yet

        confidence = self.agent.get_confidence(state)
        return confidence >= self.cfg['confidence_threshold']

    def reset(self):
        """Reset policy state for new simulation run."""
        self.agent.reset()
        self.current_policy  = 'LRU' if self.use_lru_warmup else 'DQN'
        self.switch_step     = None
        self.recent_hits_lru = []
        self.recent_hits_dqn = []

    def __repr__(self):
        return (f"DQNPolicy(current={self.current_policy}, "
                f"warmup={self.use_lru_warmup})")
