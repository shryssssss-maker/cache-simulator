# policies/fifo.py
# =============================================================================
# FIFO — First In, First Out
# Evicts the block that was loaded into cache longest ago (highest age)
# =============================================================================


class FIFOPolicy:
    """
    First-In First-Out replacement policy.
    Evicts the oldest block (by age — steps since loaded into cache).
    Does not consider recency or frequency of access.
    Simple but performs poorly on repeated access patterns.
    """

    def __init__(self):
        self.last_was_override = False

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select block to evict: highest age wins (loaded longest ago).

        Args:
            cache_set:    CacheSet containing blocks to choose from
            step:         current simulation step (unused by FIFO)
            incoming_tag: tag of incoming block (unused by FIFO)

        Returns:
            CacheBlock to evict
        """
        self.last_was_override = False
        return max(cache_set.blocks, key=lambda b: b.age)

    def reset(self):
        pass

    def __repr__(self):
        return "FIFOPolicy"