import unittest
import sys
sys.path.insert(0, '..')
from config import CONFIG
from cache.simulator import CacheSimulator
from policies.lru import LRUPolicy
from policies.fifo import FIFOPolicy
from policies.belady import BeladyPolicy
from cache.decoder import AddressDecoder
from data.generator import generate_trace, split_trace, extract_tags


class TestLRUPolicy(unittest.TestCase):

    def _make_trace(self, ops_and_addrs):
        """Convert simple tag list to (op, address) trace."""
        decoder = AddressDecoder(CONFIG)
        return [('R', tag * CONFIG['block_size'])
                for tag in ops_and_addrs]

    def test_basic_lru_eviction_no_hits(self):
        """
        Cache 2-way, trace A B C A
        Step 1: A -> MISS [A]
        Step 2: B -> MISS [A,B] (full)
        Step 3: C -> MISS evict LRU=A -> [B,C]
        Step 4: A -> MISS evict LRU=B -> [C,A]
        Expected: 0 hits, 4 misses
        """
        import importlib, config
        cfg          = dict(CONFIG)
        cfg['associativity'] = 2
        cfg['cache_size']    = 128
        cfg['num_sets']      = 1
        cfg['state_size']    = 8
        cfg['action_size']   = 2

        sim   = CacheSimulator(policy=LRUPolicy(), cfg=cfg)
        trace = [('R', 0), ('R', 64), ('R', 128), ('R', 0)]
        sim.run_trace(trace)
        self.assertEqual(sim.metrics.hits,   0)
        self.assertEqual(sim.metrics.misses, 4)

    def test_lru_hit_pattern(self):
        """
        Trace A B A B A B -> 4 hits, 2 misses for 2-way cache
        """
        cfg = dict(CONFIG)
        cfg['associativity'] = 2
        cfg['cache_size']    = 128
        cfg['num_sets']      = 1
        cfg['state_size']    = 8
        cfg['action_size']   = 2

        sim   = CacheSimulator(policy=LRUPolicy(), cfg=cfg)
        trace = [('R', 0), ('R', 64), ('R', 0),
                 ('R', 64), ('R', 0), ('R', 64)]
        sim.run_trace(trace)
        self.assertEqual(sim.metrics.hits,   4)
        self.assertEqual(sim.metrics.misses, 2)


class TestBeladyPolicy(unittest.TestCase):

    def test_belady_beats_or_matches_lru(self):
        """Belady must always have >= hit rate than LRU."""
        decoder = AddressDecoder(CONFIG)
        trace   = generate_trace('zipfian', seed=42)
        _, test = split_trace(trace)

        # Belady
        belady    = BeladyPolicy()
        tag_trace = extract_tags(trace, decoder)
        belady.precompute(tag_trace)
        sim_b     = CacheSimulator(policy=belady)
        sim_b.run_trace(test)

        # LRU
        sim_lru = CacheSimulator(policy=LRUPolicy())
        sim_lru.run_trace(test)

        self.assertGreaterEqual(
            sim_b.metrics.hit_rate,
            sim_lru.metrics.hit_rate - 0.001,   # small tolerance
            "Belady must be >= LRU"
        )


class TestMetrics(unittest.TestCase):

    def test_hits_plus_misses_equals_total(self):
        sim   = CacheSimulator(policy=LRUPolicy())
        trace = generate_trace('random', seed=42)
        _, test = split_trace(trace)
        sim.run_trace(test)
        self.assertEqual(
            sim.metrics.hits + sim.metrics.misses,
            sim.metrics.total_accesses
        )

    def test_hit_rate_between_zero_and_one(self):
        sim   = CacheSimulator(policy=LRUPolicy())
        trace = generate_trace('random', seed=42)
        _, test = split_trace(trace)
        sim.run_trace(test)
        self.assertGreaterEqual(sim.metrics.hit_rate, 0.0)
        self.assertLessEqual(sim.metrics.hit_rate,    1.0)

    def test_readonly_trace_zero_writebacks(self):
        """Read-only trace must have zero writebacks."""
        trace = [('R', i * 64) for i in range(1000)]
        sim   = CacheSimulator(policy=LRUPolicy())
        sim.run_trace(trace)
        self.assertEqual(sim.metrics.writebacks, 0)

    def test_amat_positive(self):
        sim   = CacheSimulator(policy=LRUPolicy())
        trace = generate_trace('random', seed=42)
        _, test = split_trace(trace)
        sim.run_trace(test)
        self.assertGreater(sim.metrics.amat, 0)

    def test_writeback_rate_between_zero_and_one(self):
        sim   = CacheSimulator(policy=LRUPolicy())
        trace = generate_trace('random', seed=42)
        _, test = split_trace(trace)
        sim.run_trace(test)
        self.assertGreaterEqual(sim.metrics.writeback_rate, 0.0)
        self.assertLessEqual(sim.metrics.writeback_rate,    1.0)
