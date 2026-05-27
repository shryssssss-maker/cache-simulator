# cache/simulator.py
# =============================================================================
# CACHE SIMULATOR CORE
# Manages cache sets, performs lookups, calls replacement policy
# =============================================================================

import numpy as np
from config import CONFIG
from cache.block import CacheBlock
from cache.ram import RAM
from cache.decoder import AddressDecoder


class Metrics:
    def __init__(self):
        self.hits            = 0
        self.misses          = 0
        self.writebacks      = 0
        self.evictions       = 0
        self.dirty_evictions = 0
        self.safety_overrides = 0
        self.total_accesses  = 0

    @property
    def hit_rate(self):
        if self.total_accesses == 0:
            return 0.0
        return self.hits / self.total_accesses

    @property
    def miss_rate(self):
        return 1.0 - self.hit_rate

    @property
    def writeback_rate(self):
        if self.evictions == 0:
            return 0.0
        return self.dirty_evictions / self.evictions

    @property
    def amat(self):
        return (
            CONFIG['hit_cycles']
            + (self.miss_rate * CONFIG['miss_cycles'])
            + (self.writeback_rate * CONFIG['writeback_cycles'])
        )

    def record_hit(self):
        self.hits += 1
        self.total_accesses += 1

    def record_miss(self):
        self.misses += 1
        self.total_accesses += 1

    def record_writeback(self):
        self.writebacks += 1
        self.dirty_evictions += 1
        self.evictions += 1

    def record_eviction(self, was_dirty):
        self.evictions += 1
        if was_dirty:
            self.dirty_evictions += 1

    def record_safety_override(self):
        self.safety_overrides += 1

    def summary(self):
        return {
            'hits'            : self.hits,
            'misses'          : self.misses,
            'total_accesses'  : self.total_accesses,
            'hit_rate'        : round(self.hit_rate, 4),
            'miss_rate'       : round(self.miss_rate, 4),
            'writeback_rate'  : round(self.writeback_rate, 4),
            'amat'            : round(self.amat, 2),
            'writebacks'      : self.writebacks,
            'evictions'       : self.evictions,
            'safety_overrides': self.safety_overrides,
        }

    def __repr__(self):
        return (f"Metrics(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.3f}, amat={self.amat:.1f})")


class CacheSet:
    def __init__(self, capacity=None):
        if capacity is None:
            capacity = CONFIG['associativity']
        self.capacity = capacity
        self.blocks   = []

    @property
    def is_full(self):
        return len(self.blocks) >= self.capacity

    def find_block(self, tag):
        for block in self.blocks:
            if block.tag == tag:
                return block
        return None

    def add_block(self, tag, ram):
        data  = ram.read(tag)
        block = CacheBlock(tag)
        block.data = data
        self.blocks.append(block)
        return block

    def remove_block(self, block):
        self.blocks.remove(block)

    def tick_all(self):
        for block in self.blocks:
            block.tick()

    def get_state(self, max_val=None):
        if max_val is None:
            max_val = CONFIG['trace_length']
            
        # Sort blocks to enforce permutation-invariance for the MLP!
        # Lowest frequency first. On ties, highest recency (oldest) first.
        self.blocks.sort(key=lambda b: (b.frequency, -b.recency))
        
        state = []
        for block in self.blocks:
            state.extend(block.to_state(max_val))
        while len(state) < CONFIG['state_size']:
            state.append(0.0)
        return np.array(state, dtype=np.float32)

    def __repr__(self):
        return f"CacheSet({len(self.blocks)}/{self.capacity} blocks)"


class CacheSimulator:
    def __init__(self, policy, cfg=CONFIG):
        self.cfg     = cfg
        self.policy  = policy
        self.ram     = RAM()
        self.decoder = AddressDecoder(cfg)
        self.metrics = Metrics()
        self.num_sets = cfg['num_sets']
        self.sets = [CacheSet(cfg['associativity'])
                     for _ in range(max(self.num_sets, 1))]
        self.step = 0

    def access(self, op, address):
        self.step += 1
        tag, index, offset = self.decoder.decode(address)
        cache_set = self.sets[index]
        block = cache_set.find_block(tag)

        result = {
            'hit'           : False,
            'evicted_block' : None,
            'loaded_tag'    : None,
            'tag'           : tag,
            'index'         : index,
        }

        if block is not None:
            if op == 'R':
                block.read()
            else:
                block.write()
            self.metrics.record_hit()
            result['hit'] = True
        else:
            self.metrics.record_miss()
            if cache_set.is_full:
                evicted_block = self._evict(cache_set, tag)
                result['evicted_block'] = evicted_block
            new_block = cache_set.add_block(tag, self.ram)
            if op == 'W':
                new_block.write()
            result['loaded_tag'] = tag

        cache_set.tick_all()
        return result

    def _evict(self, cache_set, incoming_tag):
        chosen_block = self.policy.select_eviction(
            cache_set, self.step - 1, incoming_tag
        )
        if chosen_block.dirty:
            self.ram.writeback(chosen_block)
            self.metrics.record_writeback()
        else:
            self.metrics.record_eviction(False)
        if hasattr(self.policy, 'last_was_override') \
                and self.policy.last_was_override:
            self.metrics.record_safety_override()
        cache_set.remove_block(chosen_block)
        return chosen_block

    def run_trace(self, trace, start_step=0):
        self.reset(start_step=start_step)
        for op, address in trace:
            self.access(op, address)
        return self.metrics

    def reset(self, start_step=0):
        self.ram     = RAM()
        self.metrics = Metrics()
        self.sets    = [CacheSet(self.cfg['associativity'])
                        for _ in range(max(self.num_sets, 1))]
        self.step    = start_step
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def __repr__(self):
        return (f"CacheSimulator(policy={type(self.policy).__name__}, "
                f"sets={self.num_sets}, ways={self.cfg['associativity']})")


# =============================================================================
# HIGH LEVEL RUNNERS — paste these at the bottom, outside all classes
# =============================================================================

def run_simulation(policy, trace, cfg=CONFIG):
    """
    Run a single simulation and return metrics summary dict.
    """
    sim = CacheSimulator(policy=policy, cfg=cfg)
    sim.run_trace(trace)
    return sim.metrics.summary()


def run_simulation_averaged(policy_factory, trace_pattern, cfg=CONFIG):
    """
    Run simulation 3 times with different seeds, return averaged metrics.
    """
    from data.generator import generate_trace, split_trace

    all_results = []

    for seed in cfg['random_seeds']:
        trace = generate_trace(trace_pattern, seed=seed)
        _, test = split_trace(trace)
        policy = policy_factory()
        result = run_simulation(policy, test, cfg)
        all_results.append(result)

    averaged = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results
                if isinstance(r[key], (int, float))]
        if vals:
            averaged[key] = round(sum(vals) / len(vals), 4)

    return averaged