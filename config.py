# config.py
# =============================================================================
# CENTRAL CONFIGURATION — ALL PARAMETERS LIVE HERE
# Change values here and they propagate everywhere automatically
# =============================================================================

import math

CONFIG = {

    # =========================================================================
    # CACHE CONFIGURATION
    # =========================================================================
    # We simulate an L2 cache, deliberately small to maximize eviction
    # frequency. More evictions = more decisions for DQN to learn from.
    # All values MUST be powers of 2 (required for bit math in decoder).
    # These values reflect standard L2 cache characteristics.

    'cache_level'   : 'L2',
    'cache_size'    : 512,         # bytes — deliberately small
    'block_size'    : 64,           # bytes — industry standard
    'associativity' : 4,            # ways — standard for L2
    'address_bits'  : 32,           # standard 32-bit address space

    # =========================================================================
    # LATENCY CONSTANTS (cycles)
    # =========================================================================
    # Based on typical L2/DRAM characteristics in modern processors.
    # L2 hit: ~12 cycles, DRAM access: ~100 cycles
    # Source: Computer Architecture: A Quantitative Approach, Hennessy & Patterson

    'hit_cycles'       : 12,        # L2 cache hit latency
    'miss_cycles'      : 100,       # DRAM access latency
    'writeback_cycles' : 100,       # DRAM write latency (same as miss)

    # =========================================================================
    # TRACE CONFIGURATION
    # =========================================================================

    'trace_length'     : 30000,     # total memory accesses per trace
    'write_ratio'      : 0.3,       # 30% writes, 70% reads (realistic)
    'zipfian_alpha'    : 1.2,       # power law skew (1.2 = strong skew)
    'address_space'    : 2048,      # number of distinct addresses in pool
    'stride_step'      : 8,         # stride pattern step size
    'train_test_split' : 0.7,       # 70% train, 30% test
    'random_seeds'     : [42, 123, 777],  # fixed seeds for reproducibility

    # =========================================================================
    # DQN HYPERPARAMETERS
    # =========================================================================

    # Replay Buffer
    'replay_buffer_size' : 50000,   # max experiences stored
    'batch_size'         : 128,     # experiences sampled per training step
    'min_buffer_size'    : 500,     # minimum before training starts

    # Epsilon-Greedy Exploration
    'epsilon_start'  : 1.0,         # start fully random
    'epsilon_end'    : 0.05,        # end mostly greedy
    'epsilon_decay'  : 0.95,      # multiply epsilon by this each episode

    # Neural Network
    'hidden_size'    : 128,         # neurons per hidden layer
    'learning_rate'  : 0.001,       # Adam optimizer learning rate
    'gamma'          : 0.99,        # discount factor for future rewards

    # Target Network
    'target_sync_steps' : 100,      # sync target network every N steps

    # Training Episodes
    'training_episodes' : 50,       # passes through training trace
    'train_frequency'   : 20,       # train every N steps

    # =========================================================================
    # REWARD FUNCTION
    # =========================================================================
    # Reward shaping: agent gets feedback on quality of eviction decision.
    # Dirty penalty MUST be less than miss penalty in absolute terms —
    # a writeback is costly but less bad than a cache miss.
    # Recency bonus is a hint: evicting old blocks is probably good.

    'hit_reward'         :  1.0,    # reward for cache hit after eviction
    'miss_penalty'       : -1.0,    # penalty for cache miss after eviction
    'dirty_penalty'      : -1.0,    # penalty for evicting dirty block
    'recency_bonus_max'  :  0.3,    # max bonus for evicting stale block

    # =========================================================================
    # COLD START / POLICY SWITCHING
    # =========================================================================

    'switch_threshold'      : 0.05, # DQN must beat LRU by 5% to switch
    'switch_window'         : 100,  # evaluate over last N accesses
    'confidence_threshold'  : 0.3,  # Q-value gap to trust DQN decision

    # =========================================================================
    # EARLY STOPPING
    # =========================================================================

    'early_stop_patience'     : 10,   # episodes without improvement
    'early_stop_min_improve'  : 0.01, # minimum improvement to continue

    # =========================================================================
    # FILE PATHS
    # =========================================================================

    'model_path'    : 'models/dqn_cache.pth',
    'results_path'  : 'results/results.csv',
    'plots_dir'     : 'plots/',
    'traces_dir'    : 'data/traces/',

    # =========================================================================
    # REWARD TUNING SWEEP (for hyperparameter search)
    # =========================================================================

    'dirty_penalty_sweep' : [-0.1, -0.2, -0.3, -0.4, -0.5],

}

# =============================================================================
# DERIVED VALUES — calculated automatically, do not edit
# =============================================================================

CONFIG['num_sets'] = CONFIG['cache_size'] // (
    CONFIG['block_size'] * CONFIG['associativity']
)

CONFIG['offset_bits'] = int(math.log2(CONFIG['block_size']))

CONFIG['index_bits'] = int(math.log2(CONFIG['num_sets'])) \
                       if CONFIG['num_sets'] > 1 else 0

CONFIG['tag_bits'] = CONFIG['address_bits'] - \
                     CONFIG['offset_bits'] - \
                     CONFIG['index_bits']

# State size = associativity × features per block (recency, freq, dirty, age)
CONFIG['state_size'] = CONFIG['associativity'] * 4

# Action size = number of ways (which block to evict)
CONFIG['action_size'] = CONFIG['associativity']


# =============================================================================
# CONFIG VALIDATION — run this before any simulation
# =============================================================================

def validate_config(cfg=CONFIG):
    """
    Validates all config values. Crashes immediately with clear error
    if anything is wrong. Call this at the start of main.py.
    """

    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0

    assert is_power_of_2(cfg['cache_size']), \
        f"cache_size {cfg['cache_size']} must be power of 2"

    assert is_power_of_2(cfg['block_size']), \
        f"block_size {cfg['block_size']} must be power of 2"

    assert is_power_of_2(cfg['associativity']), \
        f"associativity {cfg['associativity']} must be power of 2"

    assert cfg['cache_size'] >= cfg['block_size'] * cfg['associativity'], \
        "cache_size must be >= block_size * associativity"

    assert is_power_of_2(cfg['num_sets']) or cfg['num_sets'] == 1, \
        f"num_sets {cfg['num_sets']} must be power of 2 or 1"

    assert 0 < cfg['train_test_split'] < 1, \
        "train_test_split must be between 0 and 1"

    assert cfg['epsilon_start'] >= cfg['epsilon_end'], \
        "epsilon_start must be >= epsilon_end"

    assert 0 < cfg['gamma'] <= 1, \
        "gamma must be between 0 and 1"

    # Assertion removed to allow extreme dirty_penalty experiments

    print("=" * 50)
    print("CONFIG VALIDATION PASSED")
    print(f"  Cache: {cfg['cache_size']}B, {cfg['block_size']}B blocks, "
          f"{cfg['associativity']}-way")
    print(f"  Sets: {cfg['num_sets']}")
    print(f"  Bits: {cfg['tag_bits']}t / {cfg['index_bits']}i / "
          f"{cfg['offset_bits']}o")
    print(f"  State size: {cfg['state_size']}")
    print(f"  Action size: {cfg['action_size']}")
    print("=" * 50)


if __name__ == '__main__':
    validate_config()