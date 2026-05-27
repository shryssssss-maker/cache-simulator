# policies/lru.py
# =============================================================================
# LRU — Least Recently Used
# Evicts the block that was accessed least recently (highest recency)
# =============================================================================


class LRUPolicy:
    """
    Least Recently Used replacement policy.
    Evicts the block with the highest recency (accessed longest ago).

    This is the standard baseline. A well-tuned DQN should beat this.
    If DQN cannot beat LRU, the model has not learned anything useful.

    Tie-breaking: if two blocks have equal recency, evict the one with
    lower frequency (accessed less overall).
    """

    def __init__(self):
        self.last_was_override = False

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select block to evict: highest recency wins.
        Tie-breaking: lowest frequency.

        Args:
            cache_set:    CacheSet containing blocks
            step:         unused
            incoming_tag: unused

        Returns:
            CacheBlock to evict
        """
        self.last_was_override = False
        return max(cache_set.blocks,
                   key=lambda b: (b.recency, -b.frequency))

    def reset(self):
        pass

    def __repr__(self):
        return "LRUPolicy"