# cache/block.py
# =============================================================================
# CACHE BLOCK
# Represents a single cache line with all metadata needed by DQN
# =============================================================================

from config import CONFIG


class CacheBlock:
    """
    A single cache block (cache line).

    Stores the tag (block identifier) and all metadata needed for:
    - Cache operation (dirty bit, tag matching)
    - DQN state construction (recency, frequency, dirty, age)

    The 4 DQN features per block:
        recency   → steps since last access (0 = just accessed)
        frequency → total accesses (cumulative)
        dirty     → 0 or 1 (has block been written to?)
        age       → steps since loaded into cache
    """

    def __init__(self, tag, block_size=None):
        if block_size is None:
            block_size = CONFIG['block_size']

        self.tag       = tag
        self.data      = [0] * block_size   # simulated block content
        self.dirty     = False              # clean by default
        self.recency   = 0                  # just loaded, so 0
        self.frequency = 1                  # loading counts as 1 access
        self.age       = 0                  # just arrived

    def read(self):
        """
        Process a read access to this block.
        Updates recency and frequency.
        Does NOT set dirty bit (reads don't modify data).
        """
        self.recency   = 0      # reset — just accessed
        self.frequency += 1

    def write(self, data=None):
        """
        Process a write access to this block.
        Updates recency and frequency.
        SETS dirty bit — block now differs from RAM copy.

        Args:
            data: optional new data to store (simulated)
        """
        self.dirty     = True   # mark dirty — RAM is now outdated
        self.recency   = 0      # reset — just accessed
        self.frequency += 1
        if data is not None:
            self.data  = data

    def tick(self):
        """
        Advance time by one step.
        Called every simulation step for every block in cache.
        Increments recency and age.
        """
        self.recency += 1
        self.age     += 1

    def to_state(self, max_val=None):
        """
        Convert block metadata to normalized feature vector for DQN.

        Returns list of 4 floats, all in range [0, 1]:
            [normalized_recency, normalized_frequency, dirty, normalized_age]

        Normalization: divide by max_val (trace length).
        dirty is already 0 or 1 — no normalization needed.

        Args:
            max_val: normalization divisor (use trace length)
        """
        if max_val is None:
            max_val = CONFIG['trace_length']

        return [
            min(self.recency   / max_val, 1.0),
            min(self.frequency / max_val, 1.0),
            float(self.dirty),              # already 0 or 1
            min(self.age       / max_val, 1.0),
        ]

    def __repr__(self):
        return (f"Block(tag={self.tag}, dirty={self.dirty}, "
                f"recency={self.recency}, freq={self.frequency}, "
                f"age={self.age})")
