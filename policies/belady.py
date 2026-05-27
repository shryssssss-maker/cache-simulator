# policies/belady.py
# =============================================================================
# BELADY'S OPTIMAL REPLACEMENT POLICY
# Evicts the block whose next use is furthest in the future
# This is the theoretical optimal — no policy can beat it
# =============================================================================

from collections import defaultdict
import bisect


class BeladyPolicy:
    """
    Belady's Optimal Cache Replacement.

    Requires knowing the entire future trace in advance.
    Impossible in real hardware, but achievable in simulation.
    Used as the theoretical upper bound (ceiling) for all comparisons.

    If any other policy beats Belady's → there is a bug somewhere.

    Time complexity: O(n) precompute + O(1) per eviction decision.
    (Precomputing avoids the naive O(n²) approach of scanning trace each time)

    Tie-breaking: when two blocks have equal next-use distance, evict
    the older one (higher age). Consistent and deterministic.
    """

    def __init__(self):
        self._occurrences    = {}     # {tag: sorted list of positions}
        self.trace           = []
        self.last_was_override = False

    def precompute(self, trace):
        """
        Build lookup table: for each (step, tag) pair, when is the
        next time this tag appears after this step?

        Must be called before any simulation starts.

        Args:
            trace: list of (op, address_tag) tuples
                   NOTE: pass tags not raw addresses for efficiency.
                   The simulator will extract tags first.

        Algorithm: scan backwards through trace, tracking last_seen.
        """
        n = len(trace)
        self.trace = trace

        # For each tag, store the sorted list of positions where it appears.
        # get_next_use() uses bisect to answer "when after step X does tag T appear?"
        # in O(log n) per query, O(n) total precompute — no O(n*tags) space needed.
        occurrences = defaultdict(list)
        for i, (_, tag) in enumerate(trace):
            occurrences[tag].append(i)
        self._occurrences = occurrences

    def get_next_use(self, current_step, tag):
        """
        Look up next use of a tag from current step.

        Args:
            current_step: current 0-indexed position in the trace
            tag:          block tag to look up

        Returns:
            integer position of next use strictly after current_step,
            or float('inf') if never accessed again
        """
        import bisect
        positions = self._occurrences.get(tag, [])
        # Find the first position strictly greater than current_step
        idx = bisect.bisect_right(positions, current_step)
        if idx < len(positions):
            return positions[idx]
        return float('inf')

    def select_eviction(self, cache_set, step, incoming_tag=None):
        """
        Select block to evict: evict block with furthest next use.

        Tie-breaking: if equal next-use distance, evict older block.

        Args:
            cache_set:    CacheSet containing blocks
            step:         current simulation step (index into trace)
            incoming_tag: unused (Belady knows future, doesn't need hint)

        Returns:
            CacheBlock to evict
        """
        self.last_was_override = False

        farthest_block    = None
        farthest_distance = -1
        oldest_age        = -1

        for block in cache_set.blocks:
            distance = self.get_next_use(step, block.tag)

            if distance == float('inf'):
                # Block never used again — perfect eviction candidate
                # Among never-used blocks, evict the oldest
                if farthest_distance != float('inf') or \
                        block.age > oldest_age:
                    farthest_distance = float('inf')
                    oldest_age        = block.age
                    farthest_block    = block
            elif distance > farthest_distance or \
                    (distance == farthest_distance and
                     block.age > oldest_age and
                     farthest_distance != float('inf')):
                farthest_distance = distance
                oldest_age        = block.age
                farthest_block    = block

        # Fallback — should never happen if set is non-empty
        if farthest_block is None:
            farthest_block = cache_set.blocks[0]

        return farthest_block

    def reset(self):
        """Note: does NOT clear _occurrences — that persists across runs."""
        pass

    def __repr__(self):
        return f"BeladyPolicy(unique_tags={len(self._occurrences)})"