# cache/ram.py
# =============================================================================
# SIMULATED RAM
# Handles writebacks from cache and provides data on cache misses
# =============================================================================


class RAM:
    """
    Simulates main memory (DRAM).

    In a real system RAM has billions of bytes.
    We simulate it as a dictionary: {tag: data}
    This is sufficient for tracking writeback correctness.
    """

    def __init__(self):
        self.storage   = {}         # {tag: block_data}
        self.writebacks = 0         # total writebacks performed
        self.reads      = 0         # total reads (on cache miss)

    def writeback(self, block):
        """
        Write a dirty block back to RAM.
        Called when a dirty block is evicted from cache.

        Args:
            block: CacheBlock object with dirty=True

        Returns:
            True on success
        """
        assert block.dirty, \
            f"writeback() called on clean block {block.tag} — bug!"
        self.storage[block.tag] = block.data.copy()
        self.writebacks += 1
        return True

    def read(self, tag):
        """
        Read block data from RAM on cache miss.
        If block not in RAM (first ever access), return zeros.

        Args:
            tag: block tag identifier

        Returns:
            block data (list of bytes)
        """
        self.reads += 1
        if tag in self.storage:
            return self.storage[tag].copy()
        else:
            # First access to this address — initialize to zeros
            return [0] * 64

    def reset(self):
        """Reset RAM state for new simulation run."""
        self.storage    = {}
        self.writebacks = 0
        self.reads      = 0

    def __repr__(self):
        return (f"RAM(blocks_stored={len(self.storage)}, "
                f"writebacks={self.writebacks}, reads={self.reads})") 