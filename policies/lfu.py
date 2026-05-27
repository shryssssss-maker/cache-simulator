# policies/lfu.py
# =============================================================================
# LFU — Least Frequently Used
# Evicts the block that has been accessed the fewest total times
# =============================================================================


class LFUPolicy:
    """
    Least Frequently Used replacement policy.
    Evicts the block with the lowest access frequency.

    Tie-breaking: if two blocks have equal frequency, evict the
    one with highest recency (LRU as tiebreaker).
    """

    def __init__(self):
        self.last_was_override = False

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select block to evict: lowest frequency wins.
        Tie-breaking: highest recency.
        """
        self.last_was_override = False
        return min(cache_set.blocks,
                   key=lambda b: (b.frequency, -b.recency))

    def reset(self):
        pass

    def __repr__(self):
        return "LFUPolicy"