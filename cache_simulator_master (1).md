# ADAPTIVE CACHE SIMULATOR WITH DQN-BASED REPLACEMENT POLICY
## COMPLETE MASTER DOCUMENT — EVERY STEP, EVERY LINE OF CODE, EVERY DECISION

---

> **HOW TO USE THIS DOCUMENT**
> Read it top to bottom. Do not skip sections. Every section builds on the previous one.
> When you see code — copy it exactly. When you see a decision — it has already been made for you.
> Follow this blindly and you will have a complete, working, impressive project.

---

# TABLE OF CONTENTS

1. [Project Summary](#1-project-summary)
2. [What You Are Building](#2-what-you-are-building)
3. [Final Architecture](#3-final-architecture)
4. [Folder Structure](#4-folder-structure)
5. [Config System](#5-config-system)
6. [Step-by-Step Implementation](#6-step-by-step-implementation)
   - 6.1 Environment Setup
   - 6.2 Config File
   - 6.3 Address Decoder
   - 6.4 Cache Block
   - 6.5 RAM
   - 6.6 Cache Simulator Core
   - 6.7 Replacement Policies (FIFO, LRU, LFU)
   - 6.8 Belady's Optimal Policy
   - 6.9 Trace Generator
   - 6.10 Trace Validator
   - 6.11 DQN Neural Network
   - 6.12 Replay Buffer
   - 6.13 Training Monitor
   - 6.14 DQN Agent
   - 6.15 Policy Switcher (Cold Start)
   - 6.16 Metrics Collector
   - 6.17 Simulation Runner
   - 6.18 Training Mode
   - 6.19 Evaluation Mode
   - 6.20 Hybrid Mode
   - 6.21 Visualization Dashboard
   - 6.22 Main Entry Point
7. [Complete Test Suite](#7-complete-test-suite)
8. [Running The Project](#8-running-the-project)
9. [What Results To Expect](#9-what-results-to-expect)
10. [Debugging Guide](#10-debugging-guide)
11. [Viva Preparation](#11-viva-preparation)
12. [Related Work](#12-related-work)
13. [Future Work — V2 with L1](#13-future-work--v2-with-l1)
14. [Complete Config Reference](#14-complete-config-reference)

---

# 1. PROJECT SUMMARY

## What This Project Is

An **L2 CPU cache simulator** that replaces the traditional LRU eviction policy with a **Deep Q-Network (DQN)** agent that learns which cache block to evict through trial and error.

## Why It Matters

Traditional cache replacement policies (LRU, FIFO, LFU) are static. They use hand-crafted rules and cannot adapt to changing workloads. This project demonstrates that a reinforcement learning agent can learn eviction decisions from raw cache access patterns — without any hand-crafted rules.

## What Makes It Different From Existing Work

| Feature | LeCaR (2018) | Hawkeye (2016) | This Project |
|---|---|---|---|
| ML-based | ✅ | ✅ | ✅ |
| Uses RL | ✅ | ❌ | ✅ |
| Uses DQN | ❌ | ❌ | ✅ |
| Dirty block penalty in reward | ❌ | ❌ | ✅ |
| Cold start solution | ❌ | ❌ | ✅ |
| Belady's as benchmark | ❌ | ✅ | ✅ |
| Interactive visualization | ❌ | ❌ | ✅ |
| Beginner accessible | ❌ | ❌ | ✅ |

## The Three Novelty Claims (Memorize These)

1. **Dirty block penalty**: DQN reward function penalizes evicting dirty blocks, addressing write-back overhead not modeled in LeCaR
2. **Confidence-based cold start**: System transitions from LRU to DQN only when agent is confident — solving cold start without contaminating benchmarks
3. **Gap-to-optimal visualization**: Interactive dashboard comparing all policies against Belady's theoretical optimum — no existing paper does this

---

# 2. WHAT YOU ARE BUILDING

## System Overview

```
[Memory Trace] → [Cache Simulator (L2)] → [Metrics] → [Dashboard]
                         ↕
               [Replacement Policy Engine]
               ┌─────────────────────────┐
               │  FIFO  LRU  LFU         │
               │  Belady's (optimal)     │
               │  DQN Agent (AI)         │
               └─────────────────────────┘
```

## The Cache Being Simulated

- **Type**: L2 cache (single level, deliberately simplified)
- **Size**: 256 bytes (deliberately small — more evictions = more AI learning)
- **Block size**: 64 bytes (industry standard)
- **Associativity**: 4-way set associative
- **Write policy**: Write-back with dirty bit tracking
- **Latency**: 12 cycles hit, 100 cycles miss, 100 cycles writeback

## The Four Operating Modes

```
python main.py train      → Train DQN offline, save model
python main.py eval       → Load model, run all benchmarks, save results.csv
python main.py visualize  → Load results.csv, generate all charts
python main.py hybrid     → Demonstrate live LRU→DQN transition
```

## The Five Policies Compared

1. **FIFO** — First In First Out (evict oldest loaded block)
2. **LRU** — Least Recently Used (evict block unused longest)
3. **LFU** — Least Frequently Used (evict block accessed least)
4. **Belady's** — Optimal (evict block used furthest in future) — THEORETICAL CEILING
5. **DQN** — Your AI agent — what you're building

## The Four Workload Patterns

1. **Sequential** — addresses increment linearly (array loops)
2. **Random** — addresses chosen randomly (unpredictable)
3. **Stride** — every Nth address (matrix operations)
4. **Zipfian** — power law distribution (most realistic — 80% of accesses hit 20% of addresses)

## The Four Metrics

1. **Hit Rate** = hits / total_accesses (higher is better)
2. **AMAT** = 12 + (miss_rate × 100) + (writeback_rate × 100) in cycles (lower is better)
3. **Writeback Rate** = dirty_evictions / total_evictions (lower is better)
4. **Convergence Speed** = steps until DQN matches LRU hit rate (unique metric)

---

# 3. FINAL ARCHITECTURE

## Component Map

```
config.py                    ← Single source of truth for ALL parameters

cache/
  decoder.py                 ← Converts raw addresses to tag/index/offset
  block.py                   ← CacheBlock with dirty bit, recency, frequency, age
  ram.py                     ← Simulated RAM for writebacks
  simulator.py               ← Core cache lookup logic

policies/
  fifo.py                    ← FIFO replacement
  lru.py                     ← LRU replacement
  lfu.py                     ← LFU replacement
  belady.py                  ← Optimal replacement (requires full trace)
  dqn.py                     ← DQN-based replacement

ml/
  network.py                 ← PyTorch neural network (16→64→64→4)
  replay_buffer.py           ← Experience replay storage
  agent.py                   ← DQN agent (epsilon-greedy, target network)
  monitor.py                 ← Training health tracker

data/
  generator.py               ← Trace generators (sequential/random/stride/zipfian)
  validator.py               ← Trace sanity checks
  traces/                    ← Saved trace CSV files

visualization/
  dashboard.py               ← All 6 Plotly charts + master dashboard

tests/
  test_decoder.py            ← Unit tests for AddressDecoder
  test_block.py              ← Unit tests for CacheBlock
  test_buffer.py             ← Unit tests for ReplayBuffer
  test_policies.py           ← Correctness tests for all policies
  test_pipeline.py           ← Integration tests

results/
  results.csv                ← Evaluation results (auto-generated)

models/
  dqn_cache.pth              ← Saved trained model (auto-generated)

plots/
  dashboard.html             ← Master interactive dashboard (auto-generated)
  01_hit_rate.html
  02_amat.html
  03_gap_to_optimal.html
  04_convergence.html
  05_writeback_rate.html
  06_training_health.html

main.py                      ← Entry point, mode selector
```

## Data Flow — Training Mode

```
1. Load CONFIG from config.py
2. Validate config (all power of 2)
3. Generate Zipfian trace (10,000 accesses)
4. Split 70/30 → train_trace / test_trace
5. Precompute next-use table for Belady's
6. Initialize: Cache, RAM, DQN Agent, Replay Buffer, Monitor
7. FOR episode in range(10):
     FOR each access in train_trace:
       a. Decode address → tag, index, offset
       b. Look up in cache set
       c. HIT → update recency, frequency, record hit
       d. MISS → need to load block
          - If set not full → load directly
          - If set full → eviction needed:
            * If buffer not ready → LRU evicts (cold start)
            * If buffer ready → epsilon-greedy decides
              - Random (epsilon) → random eviction
              - Greedy (1-epsilon) → DQN evicts
            * Calculate reward
            * Store experience in replay buffer
            * Every 20 steps: sample buffer, train online network
            * Every 100 steps: sync target network
       e. Tick all blocks (age/recency increment)
       f. Decay epsilon
     g. Log episode metrics to monitor
     h. Check early stopping
8. Save model → models/dqn_cache.pth
```

## Data Flow — Evaluation Mode

```
1. Load CONFIG
2. Load model from models/dqn_cache.pth
3. FOR each policy in [FIFO, LRU, LFU, Belady, DQN]:
     FOR each pattern in [sequential, random, stride, zipfian]:
       FOR each seed in [42, 123, 777]:
         a. Generate trace with this seed
         b. Split 70/30, take test portion only
         c. Run simulation (no training)
         d. Record: hit_rate, amat, writeback_rate
       e. Average results over 3 seeds
4. Save all results → results/results.csv
```

---

# 4. FOLDER STRUCTURE

Create this exact structure before writing any code:

```bash
mkdir cache_simulator
cd cache_simulator
mkdir cache policies ml data data/traces visualization tests results models plots
touch config.py main.py
touch cache/__init__.py cache/decoder.py cache/block.py cache/ram.py cache/simulator.py
touch policies/__init__.py policies/fifo.py policies/lru.py policies/lfu.py policies/belady.py policies/dqn.py
touch ml/__init__.py ml/network.py ml/replay_buffer.py ml/agent.py ml/monitor.py
touch data/__init__.py data/generator.py data/validator.py
touch visualization/__init__.py visualization/dashboard.py
touch tests/__init__.py tests/test_decoder.py tests/test_block.py tests/test_buffer.py tests/test_policies.py tests/test_pipeline.py
```

---

# 5. CONFIG SYSTEM

## Why One Config File

You have ~25 parameters scattered across the entire project. Without central config:
- Changing cache size requires editing 5 files
- Reproducing experiments is impossible
- Every parameter is justified and documented in one place

## config.py — The Complete File

```python
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
    'cache_size'    : 256,          # bytes — deliberately small
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

    'trace_length'     : 10000,     # total memory accesses per trace
    'write_ratio'      : 0.3,       # 30% writes, 70% reads (realistic)
    'zipfian_alpha'    : 1.2,       # power law skew (1.2 = strong skew)
    'address_space'    : 1024,      # number of distinct addresses in pool
    'stride_step'      : 8,         # stride pattern step size
    'train_test_split' : 0.7,       # 70% train, 30% test
    'random_seeds'     : [42, 123, 777],  # fixed seeds for reproducibility

    # =========================================================================
    # DQN HYPERPARAMETERS
    # =========================================================================

    # Replay Buffer
    'replay_buffer_size' : 10000,   # max experiences stored
    'batch_size'         : 32,      # experiences sampled per training step
    'min_buffer_size'    : 500,     # minimum before training starts

    # Epsilon-Greedy Exploration
    'epsilon_start'  : 1.0,         # start fully random
    'epsilon_end'    : 0.05,        # end mostly greedy
    'epsilon_decay'  : 0.995,       # multiply epsilon by this each step

    # Neural Network
    'hidden_size'    : 64,          # neurons per hidden layer
    'learning_rate'  : 0.001,       # Adam optimizer learning rate
    'gamma'          : 0.99,        # discount factor for future rewards

    # Target Network
    'target_sync_steps' : 100,      # sync target network every N steps

    # Training Episodes
    'training_episodes' : 10,       # passes through training trace
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
    'dirty_penalty'      : -0.2,    # penalty for evicting dirty block
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

    'early_stop_patience'     : 3,    # episodes without improvement
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

    assert abs(cfg['dirty_penalty']) < abs(cfg['miss_penalty']), \
        "dirty_penalty must be smaller magnitude than miss_penalty"

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
```

---

# 6. STEP-BY-STEP IMPLEMENTATION

Follow this section in exact order. Do not jump ahead.

---

## 6.1 Environment Setup

### Install Python
You need Python 3.9 or higher. Check with:
```bash
python --version
```

### Create Virtual Environment
```bash
cd cache_simulator
python -m venv venv

# Activate (Windows):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### Install All Dependencies
```bash
pip install torch torchvision numpy plotly pandas scipy
```

Exact versions that work together:
```bash
pip install torch==2.0.1 numpy==1.24.3 plotly==5.15.0 pandas==2.0.3 scipy==1.11.1
```

### Verify Installation
```python
# Run this in Python to verify everything works
import torch
import numpy as np
import plotly
import pandas
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"Pandas: {pandas.__version__}")
print("All good!")
```

---

## 6.2 Config File

Copy the complete `config.py` from Section 5 exactly as written.

Then verify it works:
```bash
python config.py
```

Expected output:
```
==================================================
CONFIG VALIDATION PASSED
  Cache: 256B, 64B blocks, 4-way
  Sets: 1
  Bits: 26t / 0i / 6o
  State size: 16
  Action size: 4
==================================================
```

---

## 6.3 Address Decoder

**File**: `cache/decoder.py`

**What it does**: Takes a raw 32-bit memory address and splits it into tag, index, and offset fields. This is the mathematical foundation of the cache — every address lookup starts here.

**Why each field exists**:
- **Offset** (6 bits): Which byte within the 64-byte block are you reading?
- **Index** (0 bits): Which cache set to look in? (We have 1 set, so 0 bits needed)
- **Tag** (26 bits): The "name" of this block — used to check if it's in cache

```python
# cache/decoder.py
# =============================================================================
# ADDRESS DECODER
# Converts raw memory addresses to cache tag/index/offset components
# =============================================================================

import math
from config import CONFIG


class AddressDecoder:
    """
    Splits a 32-bit memory address into tag, index, and offset.

    For our config (256B cache, 64B blocks, 4-way, 1 set):
        Offset bits = log2(64)  = 6  (which byte in block)
        Index bits  = log2(1)   = 0  (which set — only 1)
        Tag bits    = 32 - 6 - 0 = 26 (block identifier)

    Example:
        Address: 0x00001A3F (binary: 00000000000000000001101000111111)
        Offset:  6 LSBs    = 0b111111 = 63
        Index:   0 bits    = 0
        Tag:     26 MSBs   = 0b00000000000000000001101000 = 104
    """

    def __init__(self, cfg=CONFIG):
        self.block_size    = cfg['block_size']
        self.cache_size    = cfg['cache_size']
        self.associativity = cfg['associativity']
        self.address_bits  = cfg['address_bits']

        self.num_sets    = cfg['num_sets']
        self.offset_bits = cfg['offset_bits']
        self.index_bits  = cfg['index_bits']
        self.tag_bits    = cfg['tag_bits']

        # Bitmasks for extracting each field
        # Offset mask: 6 ones in LSB positions → 0b111111 = 0x3F
        self.offset_mask = (1 << self.offset_bits) - 1

        # Index mask: shifted above offset bits
        if self.index_bits > 0:
            self.index_mask = ((1 << self.index_bits) - 1) << self.offset_bits
        else:
            self.index_mask = 0

    def decode(self, address):
        """
        Decode a raw address into (tag, index, offset).

        Args:
            address: integer memory address (32-bit)

        Returns:
            (tag, index, offset) tuple of integers

        Example:
            decoder.decode(0x1A3F) → (tag=104, index=0, offset=63)
        """
        # Extract offset (bottom bits)
        offset = address & self.offset_mask

        # Extract index (middle bits)
        if self.index_bits > 0:
            index = (address & self.index_mask) >> self.offset_bits
        else:
            index = 0

        # Extract tag (top bits)
        tag = address >> (self.offset_bits + self.index_bits)

        return tag, index, offset

    def reconstruct(self, tag, index, offset):
        """
        Reconstruct original address from components.
        Used for validation — decode then reconstruct must equal original.

        Args:
            tag, index, offset: integer components

        Returns:
            reconstructed integer address
        """
        return (
            (tag << (self.offset_bits + self.index_bits)) |
            (index << self.offset_bits) |
            offset
        )

    def debug_address(self, address):
        """
        Print detailed breakdown of an address.
        Use this during development to verify decoder.
        """
        tag, index, offset = self.decode(address)
        reconstructed = self.reconstruct(tag, index, offset)

        print(f"\nAddress Breakdown:")
        print(f"  Raw address : {address:#010x} ({address})")
        print(f"  Tag         : {tag} ({tag:026b})")
        print(f"  Index       : {index} ({index:0{max(self.index_bits,1)}b})")
        print(f"  Offset      : {offset} ({offset:06b})")
        print(f"  Reconstruct : {reconstructed:#010x}")
        print(f"  Valid       : {'✅' if reconstructed == address else '❌ BUG!'}")

    def validate(self):
        """
        Run sanity checks on the decoder.
        Call this during initialization to catch config errors early.
        """
        test_addresses = [
            0x00000000,
            0xFFFFFFFF,
            0x00001A3F,
            0x00002B10,
            0x0000FFFF,
        ]

        all_passed = True
        for addr in test_addresses:
            tag, index, offset = self.decode(addr)

            # Reconstruct and verify
            reconstructed = self.reconstruct(tag, index, offset)
            if reconstructed != addr:
                print(f"❌ DECODE FAILED for {addr:#010x}")
                all_passed = False
                continue

            # Verify ranges
            if offset >= self.block_size:
                print(f"❌ OFFSET OUT OF RANGE: {offset} >= {self.block_size}")
                all_passed = False
                continue

            if index >= max(self.num_sets, 1):
                print(f"❌ INDEX OUT OF RANGE: {index} >= {self.num_sets}")
                all_passed = False
                continue

        if all_passed:
            print("AddressDecoder validation: ✅ All checks passed")
        return all_passed
```

---

## 6.4 Cache Block

**File**: `cache/block.py`

**What it does**: Represents a single block (cache line) stored in the cache. Tracks all the information needed by both the cache and the DQN agent.

**Fields**:
- `tag`: The block's identifier (from address decoder)
- `dirty`: Has this block been written to? (True = must writeback before evicting)
- `recency`: How many steps since last access (increments every tick, resets on access)
- `frequency`: Total number of times this block has been accessed
- `age`: How many steps since this block was loaded into cache

```python
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
```

---

## 6.5 RAM

**File**: `cache/ram.py`

**What it does**: Simulates main memory. When a dirty block is evicted, its data must be written back here. Also provides data when cache misses occur.

```python
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
```

---

## 6.6 Cache Simulator Core

**File**: `cache/simulator.py`

**What it does**: The heart of the project. Manages the cache sets, performs lookups, calls the replacement policy, tracks all metrics.

**Key concept — cache sets**: With 1 set and 4-way associativity, all blocks go into the same set. The set is just a list of up to 4 CacheBlocks.

```python
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
    """
    Tracks all simulation metrics.
    Reset between simulation runs.
    """

    def __init__(self):
        self.hits           = 0
        self.misses         = 0
        self.writebacks     = 0
        self.evictions      = 0
        self.dirty_evictions = 0
        self.safety_overrides = 0  # times safety guard overrode DQN
        self.total_accesses = 0

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
        """
        Average Memory Access Time in cycles.
        Formula: hit_time + (miss_rate × miss_penalty)
                          + (writeback_rate × writeback_cycles)
        """
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
    """
    A single cache set containing up to 'associativity' blocks.

    With our config: 1 set, 4 ways → 1 CacheSet with max 4 blocks.
    This is a fully-associative cache (all blocks in one set).
    """

    def __init__(self, capacity=None):
        if capacity is None:
            capacity = CONFIG['associativity']
        self.capacity = capacity
        self.blocks   = []      # list of CacheBlock objects

    @property
    def is_full(self):
        return len(self.blocks) >= self.capacity

    def find_block(self, tag):
        """
        Search for a block by tag.

        Returns:
            CacheBlock if found (cache hit), None if not found (cache miss)
        """
        for block in self.blocks:
            if block.tag == tag:
                return block
        return None

    def add_block(self, tag, ram):
        """
        Load a new block into this set.
        Fetches data from RAM.

        Args:
            tag: block tag identifier
            ram: RAM object to load data from

        Returns:
            the new CacheBlock
        """
        data  = ram.read(tag)
        block = CacheBlock(tag)
        block.data = data
        self.blocks.append(block)
        return block

    def remove_block(self, block):
        """
        Remove a block from this set.
        If dirty, writeback to RAM first.
        """
        # Caller is responsible for writeback — this just removes
        self.blocks.remove(block)

    def tick_all(self):
        """
        Advance time for all blocks in this set.
        Called every simulation step.
        """
        for block in self.blocks:
            block.tick()

    def get_state(self, max_val=None):
        """
        Build state vector for DQN from all blocks in set.
        Pads with zeros if set not full.

        Returns:
            numpy array of shape (state_size,) = (16,) for 4-way cache
            Flattened: [block0_features, block1_features, ...]
            Each block: [recency, frequency, dirty, age] — all normalized
        """
        if max_val is None:
            max_val = CONFIG['trace_length']

        state = []
        for block in self.blocks:
            state.extend(block.to_state(max_val))

        # Pad with zeros for empty ways
        while len(state) < CONFIG['state_size']:
            state.append(0.0)

        return np.array(state, dtype=np.float32)

    def __repr__(self):
        return f"CacheSet({len(self.blocks)}/{self.capacity} blocks)"


class CacheSimulator:
    """
    Main cache simulator.

    Manages cache sets, performs lookups, coordinates with
    replacement policy and metrics collector.

    Usage:
        sim = CacheSimulator(policy=LRUPolicy())
        for op, addr in trace:
            result = sim.access(op, addr)
        print(sim.metrics.summary())
    """

    def __init__(self, policy, cfg=CONFIG):
        self.cfg     = cfg
        self.policy  = policy
        self.ram     = RAM()
        self.decoder = AddressDecoder(cfg)
        self.metrics = Metrics()

        # Initialize cache sets
        self.num_sets = cfg['num_sets']
        self.sets = [CacheSet(cfg['associativity'])
                     for _ in range(max(self.num_sets, 1))]

        self.step = 0   # global step counter

    def access(self, op, address):
        """
        Process one memory access (read or write).

        Args:
            op:      'R' for read, 'W' for write
            address: integer memory address

        Returns:
            dict with keys: 'hit', 'evicted_block', 'loaded_tag'
        """
        self.step += 1

        # Step 1: Decode address
        tag, index, offset = self.decoder.decode(address)

        # Step 2: Select the correct cache set
        cache_set = self.sets[index]

        # Step 3: Search for tag in set (hit or miss?)
        block = cache_set.find_block(tag)

        result = {
            'hit'           : False,
            'evicted_block' : None,
            'loaded_tag'    : None,
            'tag'           : tag,
            'index'         : index,
        }

        if block is not None:
            # ─── CACHE HIT ───
            if op == 'R':
                block.read()
            else:
                block.write()
            self.metrics.record_hit()
            result['hit'] = True

        else:
            # ─── CACHE MISS ───
            self.metrics.record_miss()
            evicted_block = None

            if cache_set.is_full:
                # Need to evict a block first
                evicted_block = self._evict(cache_set, tag)
                result['evicted_block'] = evicted_block

            # Load new block from RAM
            new_block = cache_set.add_block(tag, self.ram)
            if op == 'W':
                new_block.write()   # immediately dirty if write miss
            result['loaded_tag'] = tag

        # Step 4: Tick all blocks (advance time)
        cache_set.tick_all()

        return result

    def _evict(self, cache_set, incoming_tag):
        """
        Select and remove a block from a full cache set.
        Handles writeback if evicted block is dirty.

        Args:
            cache_set:    the CacheSet to evict from
            incoming_tag: tag of block about to be loaded (for Belady's)

        Returns:
            the evicted CacheBlock (for experience recording in DQN)
        """
        # Ask policy to choose which block to evict
        chosen_block = self.policy.select_eviction(
            cache_set, self.step, incoming_tag
        )

        # Writeback if dirty
        if chosen_block.dirty:
            self.ram.writeback(chosen_block)
            self.metrics.record_writeback()

        # Track if safety override happened (for DQN policy)
        if hasattr(self.policy, 'last_was_override') \
                and self.policy.last_was_override:
            self.metrics.record_safety_override()

        # Remove from set
        cache_set.remove_block(chosen_block)

        return chosen_block

    def run_trace(self, trace):
        """
        Run complete trace through cache.

        Args:
            trace: list of (op, address) tuples

        Returns:
            Metrics object with all results
        """
        self.reset()
        for op, address in trace:
            self.access(op, address)
        return self.metrics

    def reset(self):
        """Reset simulator for new run."""
        self.ram     = RAM()
        self.metrics = Metrics()
        self.sets    = [CacheSet(self.cfg['associativity'])
                        for _ in range(max(self.num_sets, 1))]
        self.step    = 0
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def __repr__(self):
        return (f"CacheSimulator(policy={type(self.policy).__name__}, "
                f"sets={self.num_sets}, ways={self.cfg['associativity']})")
```

---

## 6.7 Replacement Policies — FIFO, LRU, LFU

**File**: `policies/fifo.py`

```python
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
```

**File**: `policies/lru.py`

```python
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
```

**File**: `policies/lfu.py`

```python
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
```

---

## 6.8 Belady's Optimal Policy

**File**: `policies/belady.py`

**What it does**: Uses knowledge of the full future trace to always evict the block that will be used furthest in the future. Theoretically optimal — no policy can beat it.

**Critical**: If your DQN ever beats Belady's in evaluation, there is a bug somewhere.

```python
# policies/belady.py
# =============================================================================
# BELADY'S OPTIMAL REPLACEMENT POLICY
# Evicts the block whose next use is furthest in the future
# This is the theoretical optimal — no policy can beat it
# =============================================================================


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
        self.next_use_table = {}    # {(step, tag): next_use_step}
        self.trace          = []
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
        self.next_use_table = {}

        # For each tag, track when it was last seen (scanning backwards)
        last_seen = {}  # {tag: step_index}

        for i in range(n - 1, -1, -1):
            _, tag = trace[i]  # (op, tag)

            if tag in last_seen:
                self.next_use_table[(i, tag)] = last_seen[tag]
            else:
                # This tag is never accessed again after step i
                self.next_use_table[(i, tag)] = float('inf')

            last_seen[tag] = i

    def get_next_use(self, current_step, tag):
        """
        Look up next use of a tag from current step.

        Args:
            current_step: index into trace
            tag:          block tag to look up

        Returns:
            integer step index of next use, or float('inf') if never again
        """
        return self.next_use_table.get((current_step, tag), float('inf'))

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
        """Note: does NOT clear next_use_table — that persists across runs."""
        pass

    def __repr__(self):
        return f"BeladyPolicy(table_size={len(self.next_use_table)})"
```

---

## 6.9 Trace Generator

**File**: `data/generator.py`

```python
# data/generator.py
# =============================================================================
# MEMORY TRACE GENERATOR
# Generates synthetic memory access traces for simulation
# =============================================================================

import random
import csv
import numpy as np
import os
from config import CONFIG
from cache.decoder import AddressDecoder


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _make_op(write_ratio):
    """Return 'W' or 'R' based on write ratio."""
    return 'W' if random.random() < write_ratio else 'R'


def generate_sequential(n=None, write_ratio=None, block_size=None):
    """
    Sequential access pattern.
    Simulates: array loops, linear scans.
    Addresses increment by block_size each step.

    This pattern is predictable — LRU handles it well.
    DQN should match LRU performance here.
    """
    if n is None:          n          = CONFIG['trace_length']
    if write_ratio is None: write_ratio = CONFIG['write_ratio']
    if block_size is None:  block_size  = CONFIG['block_size']

    trace = []
    for i in range(n):
        op   = _make_op(write_ratio)
        addr = (i * block_size) % (CONFIG['address_space'] * block_size)
        trace.append((op, addr))
    return trace


def generate_random(n=None, write_ratio=None):
    """
    Fully random access pattern.
    Simulates: hash tables, random memory reads.
    No pattern to learn — all policies perform similarly (poorly).

    This is the hardest workload for any replacement policy.
    """
    if n is None:          n          = CONFIG['trace_length']
    if write_ratio is None: write_ratio = CONFIG['write_ratio']

    trace = []
    for i in range(n):
        op   = _make_op(write_ratio)
        addr = random.randint(0, CONFIG['address_space'] - 1) * \
               CONFIG['block_size']
        trace.append((op, addr))
    return trace


def generate_stride(n=None, write_ratio=None, stride=None):
    """
    Stride access pattern.
    Simulates: matrix operations, image processing.
    Every Nth address is accessed.

    Moderate difficulty — some pattern but not as predictable as sequential.
    """
    if n is None:          n          = CONFIG['trace_length']
    if write_ratio is None: write_ratio = CONFIG['write_ratio']
    if stride is None:      stride      = CONFIG['stride_step']

    trace = []
    for i in range(n):
        op   = _make_op(write_ratio)
        addr = ((i * stride) % CONFIG['address_space']) * CONFIG['block_size']
        trace.append((op, addr))
    return trace


def generate_zipfian(n=None, write_ratio=None, alpha=None):
    """
    Zipfian (power law) access pattern.
    Simulates: real program behavior — 80% of accesses hit 20% of addresses.

    This is the MOST REALISTIC workload.
    DQN should show its biggest advantage here.

    The Zipfian distribution: P(address_i) ∝ 1/i^alpha
    Higher alpha = more skewed (fewer hot addresses).

    NOTE: This is a TRUE Zipfian distribution using numpy's power law
    sampling — NOT a fake 80/20 uniform split.
    """
    if n is None:          n          = CONFIG['trace_length']
    if write_ratio is None: write_ratio = CONFIG['write_ratio']
    if alpha is None:       alpha       = CONFIG['zipfian_alpha']

    address_space = CONFIG['address_space']

    # Generate proper Zipfian probabilities
    ranks = np.arange(1, address_space + 1)
    probs = 1.0 / (ranks ** alpha)
    probs /= probs.sum()    # normalize to sum to 1.0

    # Sample addresses according to Zipfian distribution
    raw_addresses = np.random.choice(address_space, size=n, p=probs)

    trace = []
    for raw_addr in raw_addresses:
        op   = _make_op(write_ratio)
        addr = int(raw_addr) * CONFIG['block_size']
        trace.append((op, addr))

    return trace


def generate_trace(pattern, seed=None, **kwargs):
    """
    Unified trace generation function.

    Args:
        pattern: 'sequential', 'random', 'stride', or 'zipfian'
        seed:    random seed for reproducibility
        **kwargs: override any default parameters

    Returns:
        list of (op, address) tuples
    """
    if seed is not None:
        set_seeds(seed)

    generators = {
        'sequential': generate_sequential,
        'random'    : generate_random,
        'stride'    : generate_stride,
        'zipfian'   : generate_zipfian,
    }

    if pattern not in generators:
        raise ValueError(f"Unknown pattern '{pattern}'. "
                         f"Choose from: {list(generators.keys())}")

    return generators[pattern](**kwargs)


def split_trace(trace, train_ratio=None):
    """
    Split trace into training and test sets.
    Always split by position (first 70% train, last 30% test).
    Never shuffle — order matters for cache simulation.

    Args:
        trace:       list of (op, address) tuples
        train_ratio: fraction for training (default 0.7)

    Returns:
        (train_trace, test_trace) tuple
    """
    if train_ratio is None:
        train_ratio = CONFIG['train_test_split']

    split_idx = int(len(trace) * train_ratio)
    return trace[:split_idx], trace[split_idx:]


def extract_tags(trace, decoder=None):
    """
    Convert trace from (op, address) to (op, tag) format.
    Needed by Belady's precompute() function.

    Args:
        trace:   list of (op, address) tuples
        decoder: AddressDecoder instance

    Returns:
        list of (op, tag) tuples
    """
    if decoder is None:
        decoder = AddressDecoder()

    return [(op, decoder.decode(addr)[0]) for op, addr in trace]


def save_trace(trace, filename):
    """
    Save trace to CSV file.

    Args:
        trace:    list of (op, address) tuples
        filename: output path
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['op', 'address'])
        writer.writerows(trace)
    print(f"Trace saved → {filename} ({len(trace)} accesses)")


def load_trace(filename):
    """
    Load trace from CSV file.

    Args:
        filename: CSV path

    Returns:
        list of (op, address) tuples
    """
    trace = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            op   = row['op']
            addr = int(row['address'])
            trace.append((op, addr))
    return trace
```

---

## 6.10 Trace Validator

**File**: `data/validator.py`

```python
# data/validator.py
# =============================================================================
# TRACE VALIDATOR
# Sanity checks for generated traces before feeding to simulator
# =============================================================================

import numpy as np
from collections import Counter
from config import CONFIG


def validate_trace(trace, pattern_name='unknown'):
    """
    Run all sanity checks on a trace.
    Prints results and returns True if valid.

    Checks:
    1. Length matches config
    2. All ops are 'R' or 'W'
    3. All addresses are non-negative integers
    4. Write ratio is approximately correct
    5. Address range is within expected bounds

    Args:
        trace:        list of (op, address) tuples
        pattern_name: name for display purposes

    Returns:
        True if all checks pass
    """
    print(f"\n--- Validating trace: {pattern_name} ---")
    errors = []

    # Check 1: Length
    if len(trace) != CONFIG['trace_length']:
        errors.append(f"Length {len(trace)} != expected {CONFIG['trace_length']}")

    # Check 2: Valid ops
    invalid_ops = [(i, op) for i, (op, _) in enumerate(trace)
                   if op not in ('R', 'W')]
    if invalid_ops:
        errors.append(f"Invalid ops at positions: {invalid_ops[:5]}")

    # Check 3: Valid addresses
    invalid_addrs = [(i, addr) for i, (_, addr) in enumerate(trace)
                     if not isinstance(addr, int) or addr < 0]
    if invalid_addrs:
        errors.append(f"Invalid addresses at: {invalid_addrs[:5]}")

    # Check 4: Write ratio
    actual_writes = sum(1 for op, _ in trace if op == 'W')
    actual_ratio  = actual_writes / len(trace)
    expected      = CONFIG['write_ratio']
    if abs(actual_ratio - expected) > 0.05:
        errors.append(f"Write ratio {actual_ratio:.2f} far from "
                      f"expected {expected:.2f}")

    # Check 5: Address range
    addresses = [addr for _, addr in trace]
    max_expected = CONFIG['address_space'] * CONFIG['block_size']
    if max(addresses) >= max_expected:
        errors.append(f"Max address {max(addresses)} >= "
                      f"expected max {max_expected}")

    # Print results
    stats = {
        'total_accesses' : len(trace),
        'unique_addresses': len(set(addr for _, addr in trace)),
        'write_ratio'    : round(actual_ratio, 3),
        'min_address'    : min(addresses),
        'max_address'    : max(addresses),
    }

    for key, val in stats.items():
        print(f"  {key:20}: {val}")

    if errors:
        for e in errors:
            print(f"  ❌ ERROR: {e}")
        return False
    else:
        print(f"  ✅ All checks passed")
        return True


def validate_zipfian(trace):
    """
    Special validation for Zipfian traces.
    Verifies that access distribution follows power law (not uniform).

    Checks:
    - Top address is accessed significantly more than median
    - Distribution is monotonically decreasing in rank order

    Args:
        trace: list of (op, address) tuples

    Returns:
        True if distribution looks Zipfian
    """
    addresses  = [addr for _, addr in trace]
    counts     = Counter(addresses).most_common()
    top_count  = counts[0][1]
    median_idx = len(counts) // 2
    median_count = counts[median_idx][1]

    skew_ratio = top_count / max(median_count, 1)

    print(f"\n--- Zipfian Distribution Check ---")
    print(f"  Total unique addresses : {len(counts)}")
    print(f"  Top address count      : {top_count}")
    print(f"  Median address count   : {median_count}")
    print(f"  Skew ratio (top/median): {skew_ratio:.1f}x")

    top_5 = counts[:5]
    print(f"  Top 5 addresses:")
    for addr, count in top_5:
        bar = '█' * (count * 40 // top_count)
        print(f"    {addr:6}: {count:4} {bar}")

    if skew_ratio < 3:
        print(f"  ⚠️  Low skew ratio — may not be truly Zipfian")
        return False
    else:
        print(f"  ✅ Good Zipfian distribution")
        return True


def validate_split(train_trace, test_trace):
    """
    Validate train/test split is reasonable.

    Checks:
    - Sizes match expected ratio
    - Test addresses appear in training (good coverage)

    Args:
        train_trace, test_trace: split trace portions

    Returns:
        True if split looks valid
    """
    total = len(train_trace) + len(test_trace)
    train_ratio = len(train_trace) / total

    train_addrs = set(addr for _, addr in train_trace)
    test_addrs  = set(addr for _, addr in test_trace)
    overlap     = len(train_addrs & test_addrs) / max(len(test_addrs), 1)

    print(f"\n--- Train/Test Split Validation ---")
    print(f"  Train size    : {len(train_trace)} ({train_ratio:.1%})")
    print(f"  Test size     : {len(test_trace)}  ({1-train_ratio:.1%})")
    print(f"  Address overlap: {overlap:.1%} of test addrs in train")

    if overlap < 0.5:
        print(f"  ⚠️  Low overlap — DQN may not have seen test addresses")
    else:
        print(f"  ✅ Good overlap")

    return True
```

---

## 6.11 DQN Neural Network

**File**: `ml/network.py`

```python
# ml/network.py
# =============================================================================
# DQN NEURAL NETWORK
# Maps cache state → Q-values for each possible eviction action
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for cache replacement decisions.

    Architecture: 16 → 64 → 64 → 4
    Input:  16 normalized features (4 blocks × 4 features each)
    Output: 4 Q-values (one per possible block to evict)

    The Q-value for action i represents:
        "How good is it to evict block i, given current cache state?"

    We evict the block with the LOWEST Q-value
    (lowest expected future reward = least valuable to keep).

    All inputs are normalized to [0, 1] before being fed to network.
    ReLU activations prevent vanishing gradients.
    No output activation — Q-values can be any real number.
    """

    def __init__(self, state_size=None, action_size=None, hidden_size=None):
        super(DQNNetwork, self).__init__()

        if state_size  is None: state_size  = CONFIG['state_size']
        if action_size is None: action_size = CONFIG['action_size']
        if hidden_size is None: hidden_size = CONFIG['hidden_size']

        self.state_size  = state_size
        self.action_size = action_size

        # Network layers
        self.fc1 = nn.Linear(state_size,  hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Initialize weights using Xavier initialization
        # This helps gradients flow properly at the start of training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        """
        Forward pass through network.

        Args:
            x: tensor of shape (batch_size, state_size) or (state_size,)

        Returns:
            tensor of shape (batch_size, action_size) — Q-values
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)         # no activation on output
        return x

    def get_action(self, state_array):
        """
        Get the greedy action (evict block with lowest Q-value).

        Args:
            state_array: numpy array of shape (state_size,)

        Returns:
            integer action index (which block to evict)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            q_values     = self.forward(state_tensor)
            # Evict block with LOWEST Q-value (least valuable to keep)
            action       = q_values.argmin(dim=1).item()
        return action

    def get_q_values(self, state_array):
        """
        Get all Q-values for a state (used for confidence check).

        Args:
            state_array: numpy array of shape (state_size,)

        Returns:
            numpy array of Q-values, shape (action_size,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
            q_values     = self.forward(state_tensor)
        return q_values.numpy().flatten()
```

---

## 6.12 Replay Buffer

**File**: `ml/replay_buffer.py`

```python
# ml/replay_buffer.py
# =============================================================================
# EXPERIENCE REPLAY BUFFER
# Stores (state, action, reward, next_state) tuples for DQN training
# =============================================================================

import random
import numpy as np
from collections import deque
from config import CONFIG


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.

    Why we need this:
    - Sequential cache accesses are highly correlated
    - Training on correlated data produces biased, unstable learning
    - Random sampling from buffer breaks correlation
    - This is the standard approach from the original DQN paper (Mnih et al. 2015)

    Capacity: 10,000 experiences
    - Too small (100): forgets old experiences, overfits to recent
    - Too large (1M): slow, needs more RAM, overkill for this project
    - 10,000: sweet spot for a trace of 10,000 accesses

    Min size before training: 500
    - Don't train on almost-empty buffer
    - Need enough diversity to sample meaningful batches
    """

    def __init__(self, capacity=None, min_size=None):
        if capacity is None: capacity = CONFIG['replay_buffer_size']
        if min_size is None: min_size = CONFIG['min_buffer_size']

        self.capacity = capacity
        self.min_size = min_size

        # deque automatically drops oldest when maxlen exceeded
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done=False):
        """
        Store one experience tuple.

        Args:
            state:      numpy array — cache state before eviction
            action:     int — which block was evicted (0 to ways-1)
            reward:     float — reward received
            next_state: numpy array — cache state after loading new block
            done:       bool — end of episode (unused but standard DQN API)
        """
        experience = (
            np.array(state,      dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            done,
        )
        self.buffer.append(experience)

    def sample(self, batch_size=None):
        """
        Sample a random batch of experiences.
        Only call when buffer is ready (len >= min_size).

        Args:
            batch_size: number of experiences to sample

        Returns:
            tuple of (states, actions, rewards, next_states, dones)
            each as numpy arrays
        """
        if batch_size is None:
            batch_size = CONFIG['batch_size']

        batch = random.sample(self.buffer, batch_size)

        states     = np.array([e[0] for e in batch], dtype=np.float32)
        actions    = np.array([e[1] for e in batch], dtype=np.int64)
        rewards    = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones      = np.array([e[4] for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def ready(self, min_size=None):
        """
        Check if buffer has enough experiences to start training.

        Returns:
            True if len(buffer) >= min_size
        """
        if min_size is None:
            min_size = self.min_size
        return len(self.buffer) >= min_size

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return (f"ReplayBuffer(size={len(self.buffer)}/{self.capacity}, "
                f"ready={self.ready()})")
```

---

## 6.13 Training Monitor

**File**: `ml/monitor.py`

```python
# ml/monitor.py
# =============================================================================
# TRAINING MONITOR
# Tracks training health metrics — convergence, epsilon decay, safety overrides
# =============================================================================

from config import CONFIG


class TrainingMonitor:
    """
    Monitors DQN training progress.

    Tracks per-episode metrics to:
    1. Detect if training is converging (hit rate improving)
    2. Support early stopping
    3. Generate convergence visualization chart
    4. Track safety guard usage (should decrease over training)

    Also stores LRU baseline for comparison on convergence chart.
    """

    def __init__(self):
        self.episode_hits       = []    # hit rate per episode
        self.episode_rewards    = []    # total reward per episode
        self.epsilon_history    = []    # epsilon at end of each episode
        self.loss_history       = []    # average loss per episode
        self.safety_overrides   = []    # safety overrides per episode

        self.lru_baseline    = None     # set externally after LRU eval
        self.switch_episode  = None     # episode when DQN took over

        self.best_hit_rate   = 0.0
        self.episodes_without_improvement = 0

    def log_episode(self, hit_rate, total_reward, epsilon,
                    avg_loss, overrides):
        """
        Log metrics for one completed training episode.

        Args:
            hit_rate:     float — hit rate this episode
            total_reward: float — sum of all rewards this episode
            epsilon:      float — current epsilon value
            avg_loss:     float — average training loss this episode
            overrides:    int   — number of safety guard overrides
        """
        self.episode_hits.append(hit_rate)
        self.episode_rewards.append(total_reward)
        self.epsilon_history.append(epsilon)
        self.loss_history.append(avg_loss)
        self.safety_overrides.append(overrides)

        episode_num = len(self.episode_hits)
        print(f"Episode {episode_num:2d} | "
              f"HitRate={hit_rate:.3f} | "
              f"Reward={total_reward:.1f} | "
              f"ε={epsilon:.3f} | "
              f"Loss={avg_loss:.4f} | "
              f"Overrides={overrides}")

    def is_learning(self):
        """
        Check if hit rate is trending upward over last 3 episodes.

        Returns:
            True  → improving
            None  → plateau
            False → getting worse
        """
        if len(self.episode_hits) < 3:
            return None     # too early to judge

        early  = sum(self.episode_hits[:3])  / 3
        recent = sum(self.episode_hits[-3:]) / 3

        if recent > early + 0.005:
            return True
        elif recent < early - 0.005:
            return False
        else:
            return None     # plateau

    def should_early_stop(self):
        """
        Check if training should stop early.
        Stops when hit rate hasn't improved by min_improve
        for patience consecutive episodes.

        Returns:
            True if training should stop
        """
        if len(self.episode_hits) < CONFIG['early_stop_patience'] + 1:
            return False

        current_best = max(self.episode_hits)
        recent_best  = max(self.episode_hits[-CONFIG['early_stop_patience']:])

        if current_best > self.best_hit_rate + CONFIG['early_stop_min_improve']:
            self.best_hit_rate = current_best
            self.episodes_without_improvement = 0
            return False
        else:
            self.episodes_without_improvement += 1
            if self.episodes_without_improvement >= \
                    CONFIG['early_stop_patience']:
                print(f"\nEarly stopping: no improvement for "
                      f"{CONFIG['early_stop_patience']} episodes")
                return True
            return False

    def safety_guard_decreasing(self):
        """
        Check if safety overrides are decreasing — good sign that
        DQN is learning to avoid dirty evictions naturally.
        """
        if len(self.safety_overrides) < 5:
            return None
        early  = sum(self.safety_overrides[:3]) / 3
        recent = sum(self.safety_overrides[-3:]) / 3
        return recent < early

    def get_summary(self):
        """Return summary dict for reporting."""
        if not self.episode_hits:
            return {}
        return {
            'episodes'          : len(self.episode_hits),
            'final_hit_rate'    : self.episode_hits[-1],
            'best_hit_rate'     : max(self.episode_hits),
            'final_epsilon'     : self.epsilon_history[-1]
                                  if self.epsilon_history else None,
            'lru_baseline'      : self.lru_baseline,
            'switch_episode'    : self.switch_episode,
            'safety_guard_ok'   : self.safety_guard_decreasing(),
        }
```

---

## 6.14 DQN Agent

**File**: `ml/agent.py`

This is the most complex file. Read carefully.

```python
# ml/agent.py
# =============================================================================
# DQN AGENT
# Combines network, replay buffer, epsilon-greedy, target network
# into a complete learning agent
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
from config import CONFIG
from ml.network import DQNNetwork
from ml.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Complete DQN agent for cache replacement.

    Components:
    1. Online network    → actively trained, makes decisions
    2. Target network    → frozen copy, provides stable training targets
    3. Replay buffer     → stores experiences, breaks correlation
    4. Epsilon-greedy    → balances exploration vs exploitation
    5. Safety guard      → prevents dirty evictions when clean available

    Training loop (called externally every N steps):
        sample batch from replay buffer
        compute target Q-values using target network
        compute current Q-values using online network
        compute loss (MSE between current and target)
        backpropagate and update online network
        periodically sync target network

    Eviction decision:
        with probability epsilon → random action (explore)
        with probability 1-epsilon → lowest Q-value action (exploit)
        safety guard → if DQN picks dirty block but clean available, override
    """

    def __init__(self, cfg=CONFIG):
        self.cfg = cfg

        # Two networks: online (learning) and target (stable reference)
        self.online_network = DQNNetwork(
            cfg['state_size'],
            cfg['action_size'],
            cfg['hidden_size']
        )
        self.target_network = copy.deepcopy(self.online_network)

        # Target network is NOT trained — only updated by copying online
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer for online network only
        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=cfg['learning_rate']
        )

        self.replay_buffer  = ReplayBuffer()
        self.epsilon        = cfg['epsilon_start']

        self.step_count     = 0
        self.train_count    = 0
        self.total_loss     = 0.0
        self.loss_steps     = 0

        # For tracking safety guard usage
        self.last_was_override = False

        # For policy switcher compatibility
        self._last_state      = None
        self._last_action     = None
        self._last_evicted    = None

    # =========================================================================
    # EVICTION DECISION
    # =========================================================================

    def select_eviction(self, cache_set, step=None, incoming_tag=None):
        """
        Select which block to evict from a full cache set.

        This is called by CacheSimulator._evict() — same interface
        as all other policies (FIFO, LRU, etc.)

        Decision process:
        1. Build state from cache set
        2. Epsilon-greedy: random or DQN
        3. Safety guard: override if DQN picks dirty when clean available
        4. Store state and action for later reward assignment

        Args:
            cache_set:    CacheSet object with blocks to choose from
            step:         current simulation step
            incoming_tag: unused (DQN doesn't need future info)

        Returns:
            CacheBlock to evict
        """
        self.step_count += 1
        self.last_was_override = False

        # Build normalized state vector
        state = cache_set.get_state(max_val=self.cfg['trace_length'])

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, len(cache_set.blocks) - 1)
        else:
            # Exploit: use DQN
            q_values = self.online_network.get_q_values(state)

            # Only consider valid block indices
            n_blocks = len(cache_set.blocks)
            valid_q  = q_values[:n_blocks]
            action   = int(np.argmin(valid_q))  # lowest Q = evict this

            # Safety guard: if DQN picks dirty but clean available, override
            action = self._safety_guard(cache_set, action, q_values)

        # Store for reward assignment
        self._last_state   = state
        self._last_action  = action
        self._last_evicted = cache_set.blocks[action]

        return cache_set.blocks[action]

    def _safety_guard(self, cache_set, dqn_action, q_values):
        """
        Safety guard: prevent DQN from choosing dirty eviction when
        a clean block is available.

        During early training, DQN has random Q-values and might
        accidentally prefer dirty blocks. This guard corrects that
        until DQN learns the -0.2 dirty penalty itself.

        When safety guard fires less often over training → DQN learned ✅
        When safety guard always fires → DQN never learned ❌

        Args:
            cache_set:  CacheSet
            dqn_action: action DQN wants to take
            q_values:   all Q-values from DQN

        Returns:
            corrected action index
        """
        chosen_block = cache_set.blocks[dqn_action]

        if not chosen_block.dirty:
            # DQN already picked clean — no override needed
            return dqn_action

        # DQN picked dirty — find clean alternatives
        clean_indices = [i for i, b in enumerate(cache_set.blocks)
                         if not b.dirty]

        if not clean_indices:
            # All blocks are dirty — no override possible
            return dqn_action

        # Override: pick clean block with lowest Q-value
        self.last_was_override = True
        clean_q_values = [(i, q_values[i]) for i in clean_indices]
        best_clean     = min(clean_q_values, key=lambda x: x[1])
        return best_clean[0]

    # =========================================================================
    # REWARD AND LEARNING
    # =========================================================================

    def record_reward(self, next_cache_state, access_result):
        """
        Calculate reward and store experience in replay buffer.

        Call this AFTER the eviction and the NEXT access have been processed.
        This gives us the consequence of the eviction decision.

        Reward structure:
            +1.0  → next access was a HIT  (good eviction)
            -1.0  → next access was a MISS (bad eviction)
            -0.2  → evicted block was dirty (writeback penalty)
            +0-0.3 → recency bonus (evicting old blocks is probably good)

        Args:
            next_cache_state: CacheSet state after loading new block
            access_result:    dict from CacheSimulator.access()
        """
        if self._last_state is None:
            return  # no eviction happened yet

        # Calculate reward
        reward = self._calculate_reward(access_result)

        # Build next state
        next_state = np.array(next_cache_state, dtype=np.float32)

        # Store experience
        self.replay_buffer.push(
            self._last_state,
            self._last_action,
            reward,
            next_state,
        )

        # Reset tracking
        self._last_state   = None
        self._last_action  = None

    def _calculate_reward(self, next_result):
        """
        Calculate reward for the previous eviction decision.

        Args:
            next_result: result dict from the access AFTER eviction

        Returns:
            float reward
        """
        reward = 0.0

        # Primary signal: was the next access a hit or miss?
        if next_result.get('hit', False):
            reward += self.cfg['hit_reward']      # +1.0
        else:
            reward += self.cfg['miss_penalty']    # -1.0

        # Dirty penalty: was the evicted block dirty?
        if self._last_evicted and self._last_evicted.dirty:
            reward += self.cfg['dirty_penalty']   # -0.2

        # Recency bonus: reward evicting blocks that haven't been used recently
        if self._last_evicted:
            max_recency  = self.cfg['trace_length']
            recency_norm = min(self._last_evicted.recency / max_recency, 1.0)
            bonus        = recency_norm * self.cfg['recency_bonus_max']
            reward      += bonus                  # 0 to +0.3

        return reward

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train_step(self):
        """
        Perform one training step: sample batch, compute loss, update network.

        Call this every N simulation steps (N = train_frequency from config).
        Only trains if buffer has enough experiences.

        Training algorithm (DQN with target network):
        1. Sample random batch of 32 experiences
        2. For each experience:
           target_q = reward + gamma * min(target_network(next_state))
        3. Compute MSE loss between online_network(state)[action] and target_q
        4. Backpropagate, update online network weights

        Returns:
            float loss value, or None if buffer not ready
        """
        if not self.replay_buffer.ready():
            return None

        # Sample random batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample()

        # Convert to tensors
        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        # Current Q-values from online network
        # Shape: (batch_size, action_size)
        current_q_all  = self.online_network(states_t)

        # Q-values for the actions that were actually taken
        # Shape: (batch_size,)
        current_q = current_q_all.gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values from TARGET network (frozen)
        with torch.no_grad():
            next_q_all = self.target_network(next_states_t)
            # We want to EVICT, so minimize Q → use min not max
            next_q_min = next_q_all.min(dim=1)[0]

            # Bellman equation:
            # target = reward + gamma * min_next_q * (1 - done)
            target_q = rewards_t + \
                       self.cfg['gamma'] * next_q_min * (1 - dones_t)

        # Compute MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), max_norm=1.0
        )

        self.optimizer.step()

        # Track loss
        loss_val        = loss.item()
        self.total_loss += loss_val
        self.loss_steps += 1
        self.train_count += 1

        # Sync target network periodically
        if self.train_count % self.cfg['target_sync_steps'] == 0:
            self._sync_target_network()

        return loss_val

    def _sync_target_network(self):
        """
        Copy online network weights to target network.
        Called every target_sync_steps training steps.
        """
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def decay_epsilon(self):
        """
        Decay epsilon by one step.
        Call this after each simulation step.
        """
        self.epsilon = max(
            self.cfg['epsilon_end'],
            self.epsilon * self.cfg['epsilon_decay']
        )

    def get_avg_loss(self):
        """Get average loss since last call. Resets counter."""
        if self.loss_steps == 0:
            return 0.0
        avg = self.total_loss / self.loss_steps
        self.total_loss = 0.0
        self.loss_steps = 0
        return avg

    # =========================================================================
    # CONFIDENCE CHECK (for policy switcher)
    # =========================================================================

    def get_confidence(self, state):
        """
        Measure agent confidence: gap between best and second-best Q-value.
        High gap → agent is sure about its decision.
        Low gap  → agent is uncertain, better to use LRU.

        Args:
            state: numpy array of state features

        Returns:
            float confidence score
        """
        q_values = self.online_network.get_q_values(state)
        sorted_q = sorted(q_values)     # ascending
        # Confidence = gap between 2nd lowest and lowest
        # (difference between best eviction and second-best eviction)
        if len(sorted_q) >= 2:
            return sorted_q[1] - sorted_q[0]
        return 0.0

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path=None):
        """Save trained model to disk."""
        if path is None:
            path = self.cfg['model_path']
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'online_state_dict' : self.online_network.state_dict(),
            'target_state_dict' : self.target_network.state_dict(),
            'optimizer_state'   : self.optimizer.state_dict(),
            'epsilon'           : self.epsilon,
            'step_count'        : self.step_count,
            'train_count'       : self.train_count,
        }, path)
        print(f"Model saved → {path}")

    def load(self, path=None):
        """Load trained model from disk."""
        if path is None:
            path = self.cfg['model_path']
        checkpoint = torch.load(path, map_location='cpu')
        self.online_network.load_state_dict(
            checkpoint['online_state_dict']
        )
        self.target_network.load_state_dict(
            checkpoint['target_state_dict']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon    = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.train_count = checkpoint['train_count']
        self.online_network.eval()
        self.target_network.eval()
        print(f"Model loaded ← {path}")

    def reset(self):
        """Reset agent for new simulation (keeps learned weights)."""
        self._last_state   = None
        self._last_action  = None
        self._last_evicted = None
```

---

## 6.15 Policy Switcher (Cold Start Solution)

**File**: `policies/dqn.py`

```python
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
    3. When DQN confidence exceeds threshold → switch to DQN
    4. If DQN performance drops below LRU → can revert

    For clean benchmarking (Experiment A):
    Use pre-trained DQN only (no LRU fallback).
    Set use_lru_warmup=False.

    For real-world demo (Experiment B / Hybrid mode):
    Use LRU warmup → DQN transition.
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
            # Use DQN with confidence check
            confidence = self.agent.get_confidence(state)

            if confidence >= self.cfg['confidence_threshold']:
                # Agent is confident — use DQN decision
                block = self.agent.select_eviction(
                    cache_set, step, incoming_tag
                )
                self.last_was_override = self.agent.last_was_override
                return block
            else:
                # Agent uncertain — fall back to LRU
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
```

---

## 6.16 Metrics Collector

Already built into `CacheSimulator` as the `Metrics` class in section 6.6. No separate file needed.

---

## 6.17 Simulation Runner

**File**: `cache/simulator.py` already contains `run_trace()`. We add a higher-level runner here.

Add this function to the bottom of `cache/simulator.py`:

```python
# Add to cache/simulator.py after the CacheSimulator class

def run_simulation(policy, trace, cfg=CONFIG):
    """
    High-level simulation runner.
    Creates simulator, runs trace, returns metrics summary.

    Args:
        policy: any policy object (FIFO, LRU, LFU, Belady, DQN)
        trace:  list of (op, address) tuples (TEST portion only)
        cfg:    config dict

    Returns:
        dict of metrics
    """
    sim = CacheSimulator(policy=policy, cfg=cfg)
    sim.run_trace(trace)
    return sim.metrics.summary()


def run_simulation_averaged(policy_factory, trace_pattern, cfg=CONFIG):
    """
    Run simulation 3 times with different seeds, return averaged metrics.

    Args:
        policy_factory: callable that returns a fresh policy instance
        trace_pattern:  'sequential', 'random', 'stride', or 'zipfian'
        cfg:            config dict

    Returns:
        dict of averaged metrics
    """
    from data.generator import generate_trace, split_trace

    all_results = []

    for seed in cfg['random_seeds']:
        trace = generate_trace(trace_pattern, seed=seed)
        _, test = split_trace(trace)

        policy = policy_factory()   # fresh policy each run
        result = run_simulation(policy, test, cfg)
        all_results.append(result)

    # Average all numeric metrics
    averaged = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results if isinstance(r[key], (int, float))]
        if vals:
            averaged[key] = round(sum(vals) / len(vals), 4)

    return averaged
```

---

## 6.18 Training Mode

**File**: Part of `main.py` (written in section 6.22)

Here is the complete training logic as a standalone function:

```python
# training_mode.py
# =============================================================================
# TRAINING MODE
# Trains DQN offline on Zipfian trace, saves model
# =============================================================================

import random
import numpy as np
from config import CONFIG, validate_config
from cache.simulator import CacheSimulator
from cache.decoder import AddressDecoder
from data.generator import generate_trace, split_trace, extract_tags
from data.validator import validate_trace, validate_zipfian, validate_split
from policies.belady import BeladyPolicy
from ml.agent import DQNAgent
from ml.monitor import TrainingMonitor


def run_training_mode(cfg=CONFIG):
    """
    Complete DQN training pipeline.

    Steps:
    1. Generate and validate Zipfian training trace
    2. Split into train/test
    3. Precompute Belady's next-use table (for future reference)
    4. Run 10 training episodes
    5. Early stopping if converged
    6. Save model

    Args:
        cfg: config dict

    Returns:
        TrainingMonitor with training history
    """
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)

    validate_config(cfg)

    # Step 1: Generate trace
    print("\n[1/6] Generating Zipfian training trace...")
    trace = generate_trace('zipfian', seed=cfg['random_seeds'][0])
    validate_trace(trace, 'zipfian')
    validate_zipfian(trace)

    # Step 2: Split
    print("\n[2/6] Splitting train/test...")
    train_trace, test_trace = split_trace(trace)
    validate_split(train_trace, test_trace)

    # Step 3: Prepare Belady (for potential future use)
    print("\n[3/6] Precomputing Belady next-use table...")
    decoder      = AddressDecoder(cfg)
    train_tags   = extract_tags(train_trace, decoder)
    belady_policy = BeladyPolicy()
    belady_policy.precompute(train_tags)
    print(f"  Table size: {len(belady_policy.next_use_table)} entries")

    # Step 4: Initialize agent and monitor
    print("\n[4/6] Initializing DQN agent...")
    agent   = DQNAgent(cfg)
    monitor = TrainingMonitor()

    print(f"  State size  : {cfg['state_size']}")
    print(f"  Action size : {cfg['action_size']}")
    print(f"  Buffer size : {cfg['replay_buffer_size']}")
    print(f"  Episodes    : {cfg['training_episodes']}")

    # Step 5: Training loop
    print("\n[5/6] Training...")
    print("-" * 60)

    from policies.dqn import DQNPolicy
    dqn_policy = DQNPolicy(agent=agent, use_lru_warmup=False, cfg=cfg)

    for episode in range(cfg['training_episodes']):

        # Fresh simulator each episode
        sim = CacheSimulator(policy=dqn_policy, cfg=cfg)
        dqn_policy.reset()

        episode_reward    = 0.0
        episode_overrides = 0
        prev_result       = None
        losses            = []

        for step_idx, (op, address) in enumerate(train_trace):

            # Process access
            result = sim.access(op, address)

            # If previous step had an eviction, record reward now
            if prev_result and prev_result.get('evicted_block') is not None:
                cache_set   = sim.sets[0]
                next_state  = cache_set.get_state()
                agent.record_reward(next_state, result)

                reward = agent._calculate_reward(result) \
                         if agent._last_evicted else 0
                episode_reward += reward

            # Train every N steps
            if step_idx % cfg['train_frequency'] == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            # Decay epsilon
            agent.decay_epsilon()

            # Track overrides
            if dqn_policy.last_was_override:
                episode_overrides += 1

            prev_result = result

        # Log episode
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        monitor.log_episode(
            hit_rate    = sim.metrics.hit_rate,
            total_reward = episode_reward,
            epsilon     = agent.epsilon,
            avg_loss    = avg_loss,
            overrides   = episode_overrides,
        )

        # Check early stopping
        if monitor.should_early_stop():
            break

        # Learning status
        learning = monitor.is_learning()
        if learning is False and episode > 3:
            print(f"⚠️  Warning: hit rate declining — check debug guide")

    # Step 6: Save model
    print("\n[6/6] Saving model...")
    agent.save(cfg['model_path'])

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Episodes run    : {len(monitor.episode_hits)}")
    print(f"  Best hit rate   : {max(monitor.episode_hits):.3f}")
    print(f"  Final epsilon   : {agent.epsilon:.3f}")
    print(f"  Model saved     : {cfg['model_path']}")
    print("="*60)

    return monitor, agent
```

---

## 6.19 Evaluation Mode

```python
# evaluation_mode.py
# =============================================================================
# EVALUATION MODE
# Loads trained model, runs all policies on all workloads, saves results.csv
# =============================================================================

import csv
import os
from config import CONFIG
from cache.simulator import CacheSimulator, run_simulation
from cache.decoder import AddressDecoder
from data.generator import generate_trace, split_trace, extract_tags
from policies.fifo import FIFOPolicy
from policies.lru import LRUPolicy
from policies.lfu import LFUPolicy
from policies.belady import BeladyPolicy
from policies.dqn import DQNPolicy
from ml.agent import DQNAgent


def run_evaluation_mode(cfg=CONFIG):
    """
    Complete evaluation pipeline.

    Runs all 5 policies on all 4 workloads, 3 seeds each.
    Saves results to results/results.csv.

    Args:
        cfg: config dict

    Returns:
        nested dict: results[policy][pattern] = metrics_dict
    """
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)

    # Load trained DQN
    print("\nLoading trained DQN model...")
    agent = DQNAgent(cfg)
    agent.load(cfg['model_path'])

    results = {}
    decoder = AddressDecoder(cfg)

    policies_to_test = ['FIFO', 'LRU', 'LFU', "Belady's", 'DQN']
    patterns         = ['sequential', 'random', 'stride', 'zipfian']

    total_runs = len(policies_to_test) * len(patterns) * len(cfg['random_seeds'])
    run_count  = 0

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern.upper()} ---")

        # Precompute Belady's table for this pattern (all seeds)
        # We'll rebuild per seed below
        for policy_name in policies_to_test:
            run_results = []

            for seed in cfg['random_seeds']:
                run_count += 1
                print(f"  [{run_count}/{total_runs}] "
                      f"{policy_name:12} | {pattern:10} | seed={seed}")

                # Generate fresh trace with this seed
                trace = generate_trace(pattern, seed=seed)
                _, test_trace = split_trace(trace)

                # Build policy instance
                policy = _build_policy(
                    policy_name, agent, trace, decoder, cfg
                )

                # Run simulation
                metrics = run_simulation(policy, test_trace, cfg)
                run_results.append(metrics)

            # Average over seeds
            avg = _average_results(run_results)

            if policy_name not in results:
                results[policy_name] = {}
            results[policy_name][pattern] = avg

            print(f"    → HitRate={avg['hit_rate']:.3f} | "
                  f"AMAT={avg['amat']:.1f} | "
                  f"WritebackRate={avg['writeback_rate']:.3f}")

    # Save results
    _save_results(results, cfg['results_path'])

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"  Results saved: {cfg['results_path']}")
    print("="*60)

    return results


def _build_policy(name, agent, full_trace, decoder, cfg):
    """Build a fresh policy instance for one run."""

    if name == 'FIFO':
        return FIFOPolicy()

    elif name == 'LRU':
        return LRUPolicy()

    elif name == 'LFU':
        return LFUPolicy()

    elif name == "Belady's":
        # Belady needs the FULL trace precomputed
        belady = BeladyPolicy()
        tag_trace = extract_tags(full_trace, decoder)
        belady.precompute(tag_trace)
        return belady

    elif name == 'DQN':
        # Use trained agent, NO warm-up (clean eval)
        dqn = DQNPolicy(agent=agent, use_lru_warmup=False, cfg=cfg)
        return dqn

    else:
        raise ValueError(f"Unknown policy: {name}")


def _average_results(run_results):
    """Average numeric metrics over multiple runs."""
    averaged = {}
    keys = [k for k in run_results[0] if isinstance(run_results[0][k],
                                                      (int, float))]
    for key in keys:
        vals = [r[key] for r in run_results]
        averaged[key] = round(sum(vals) / len(vals), 4)
    return averaged


def _save_results(results, path):
    """Save nested results dict to CSV."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    rows = []
    for policy, patterns in results.items():
        for pattern, metrics in patterns.items():
            row = {'policy': policy, 'pattern': pattern}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved → {path}")


def load_results(path=None):
    """Load results from CSV back into nested dict."""
    if path is None:
        path = CONFIG['results_path']

    results = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy  = row['policy']
            pattern = row['pattern']
            if policy not in results:
                results[policy] = {}
            metrics = {k: float(v) for k, v in row.items()
                       if k not in ('policy', 'pattern')}
            results[policy][pattern] = metrics

    return results
```

---

## 6.20 Hybrid Mode

```python
# hybrid_mode.py
# =============================================================================
# HYBRID MODE — Experiment B
# Demonstrates live LRU→DQN transition on a single trace
# =============================================================================

from config import CONFIG
from cache.simulator import CacheSimulator
from data.generator import generate_trace, split_trace
from data.validator import validate_trace
from ml.agent import DQNAgent
from ml.monitor import TrainingMonitor
from policies.dqn import DQNPolicy


def run_hybrid_mode(cfg=CONFIG):
    """
    Demonstrates the LRU→DQN warm-up transition.

    Unlike training mode (which trains offline), hybrid mode
    trains the DQN LIVE during simulation while starting with LRU.

    This is Experiment B — shows practical deployment behavior.
    Results should NOT be compared directly to Experiment A results
    because the first N steps use LRU, not DQN.

    Args:
        cfg: config dict

    Returns:
        dict with transition metrics
    """
    print("\n" + "="*60)
    print("HYBRID MODE — Live LRU→DQN Transition")
    print("="*60)

    # Generate Zipfian trace
    trace = generate_trace('zipfian', seed=cfg['random_seeds'][0])
    validate_trace(trace, 'zipfian_hybrid')

    # Use FULL trace (no train/test split — this is a live demo)
    agent      = DQNAgent(cfg)
    dqn_policy = DQNPolicy(agent=agent, use_lru_warmup=True, cfg=cfg)
    sim        = CacheSimulator(policy=dqn_policy, cfg=cfg)
    monitor    = TrainingMonitor()

    print("\nRunning live simulation...")
    print("  Phase 1: LRU (until DQN confident)")
    print("  Phase 2: DQN (after confidence threshold met)")
    print()

    # Track metrics at switch point
    pre_switch_hits  = 0
    post_switch_hits = 0
    pre_switch_total = 0
    post_switch_total = 0
    switched         = False

    prev_result = None

    for step_idx, (op, address) in enumerate(trace):

        result = sim.access(op, address)

        # Record reward if previous step had eviction
        if prev_result and prev_result.get('evicted_block') is not None:
            next_state = sim.sets[0].get_state()
            agent.record_reward(next_state, result)

        # Train every N steps
        if step_idx % cfg['train_frequency'] == 0:
            agent.train_step()

        agent.decay_epsilon()

        # Track switch
        if dqn_policy.current_policy == 'DQN' and not switched:
            switched = True
            switch_step = step_idx
            print(f"  >>> SWITCHED TO DQN at step {step_idx} <<<")

        # Track hit rates before/after switch
        if not switched:
            pre_switch_total += 1
            if result['hit']:
                pre_switch_hits += 1
        else:
            post_switch_total += 1
            if result['hit']:
                post_switch_hits += 1

        prev_result = result

        # Progress update every 1000 steps
        if step_idx % 1000 == 0:
            print(f"  Step {step_idx:5d} | "
                  f"Policy={dqn_policy.current_policy:3} | "
                  f"HitRate={sim.metrics.hit_rate:.3f} | "
                  f"ε={agent.epsilon:.3f}")

    # Final report
    pre_hr  = pre_switch_hits  / max(pre_switch_total, 1)
    post_hr = post_switch_hits / max(post_switch_total, 1)

    print("\n" + "="*60)
    print("HYBRID MODE COMPLETE")
    print(f"  Switch step        : {dqn_policy.switch_step}")
    print(f"  Pre-switch hit rate : {pre_hr:.3f}  (LRU)")
    print(f"  Post-switch hit rate: {post_hr:.3f}  (DQN)")
    print(f"  Improvement        : {(post_hr - pre_hr)*100:+.1f}%")
    print(f"  Overall hit rate   : {sim.metrics.hit_rate:.3f}")
    print("="*60)

    return {
        'switch_step'    : dqn_policy.switch_step,
        'pre_switch_hr'  : pre_hr,
        'post_switch_hr' : post_hr,
        'overall_hr'     : sim.metrics.hit_rate,
        'overall_amat'   : sim.metrics.amat,
    }
```

---

## 6.21 Visualization Dashboard

**File**: `visualization/dashboard.py`

```python
# visualization/dashboard.py
# =============================================================================
# VISUALIZATION DASHBOARD
# 6 interactive Plotly charts saved as standalone HTML files
# =============================================================================

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import CONFIG


POLICY_COLORS = {
    'FIFO'     : '#E74C3C',   # red
    'LRU'      : '#E67E22',   # orange
    'LFU'      : '#F1C40F',   # yellow
    'DQN'      : '#2ECC71',   # green  ← your AI
    "Belady's" : '#3498DB',   # blue   ← optimal ceiling
}

WORKLOADS = ['sequential', 'random', 'stride', 'zipfian']
WORKLOAD_LABELS = ['Sequential', 'Random', 'Stride', 'Zipfian']


def _ensure_plots_dir(cfg=CONFIG):
    os.makedirs(cfg['plots_dir'], exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Hit Rate Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_hit_rate_comparison(results, cfg=CONFIG):
    """
    Grouped bar chart: hit rate for each policy × workload.
    Primary result chart — shows whether DQN beats LRU.
    """
    _ensure_plots_dir(cfg)
    fig = go.Figure()

    for policy, color in POLICY_COLORS.items():
        if policy not in results:
            continue
        y_vals = [
            results[policy].get(w, {}).get('hit_rate', 0) * 100
            for w in WORKLOADS
        ]
        fig.add_trace(go.Bar(
            name=policy,
            x=WORKLOAD_LABELS,
            y=y_vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in y_vals],
            textposition='outside',
        ))

    fig.update_layout(
        title={
            'text' : 'Cache Hit Rate by Policy and Workload',
            'font' : {'size': 20}
        },
        barmode     = 'group',
        yaxis_title = 'Hit Rate (%)',
        xaxis_title = 'Workload Pattern',
        yaxis_range = [0, 105],
        legend      = dict(orientation='h', y=-0.15),
        height      = 500,
        template    = 'plotly_white',
    )

    path = os.path.join(cfg['plots_dir'], '01_hit_rate.html')
    fig.write_html(path)
    print(f"Chart 1 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: AMAT Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_amat_comparison(results, cfg=CONFIG):
    """
    Grouped bar chart: AMAT (cycles) for each policy × workload.
    More complete than hit rate — includes writeback cost.
    Lower = better.
    """
    _ensure_plots_dir(cfg)
    fig = go.Figure()

    for policy, color in POLICY_COLORS.items():
        if policy not in results:
            continue
        y_vals = [
            results[policy].get(w, {}).get('amat', 0)
            for w in WORKLOADS
        ]
        fig.add_trace(go.Bar(
            name=policy,
            x=WORKLOAD_LABELS,
            y=y_vals,
            marker_color=color,
            text=[f"{v:.1f}" for v in y_vals],
            textposition='outside',
        ))

    fig.add_annotation(
        text="Lower AMAT = faster memory system",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12, color='gray')
    )

    fig.update_layout(
        title={'text': 'Average Memory Access Time (AMAT)', 'font': {'size': 20}},
        barmode     = 'group',
        yaxis_title = 'AMAT (cycles)',
        xaxis_title = 'Workload Pattern',
        legend      = dict(orientation='h', y=-0.15),
        height      = 500,
        template    = 'plotly_white',
    )

    path = os.path.join(cfg['plots_dir'], '02_amat.html')
    fig.write_html(path)
    print(f"Chart 2 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Gap to Optimal
# ─────────────────────────────────────────────────────────────────────────────

def plot_gap_to_optimal(results, cfg=CONFIG):
    """
    Line chart: how far each policy is from Belady's optimal hit rate.
    Gap = Belady's hit rate - Policy hit rate (lower = closer to optimal).

    This is the UNIQUE chart — no existing paper shows this.
    """
    _ensure_plots_dir(cfg)
    fig = go.Figure()

    if "Belady's" not in results:
        print("Warning: Belady's results not found — skipping gap chart")
        return None

    for policy, color in POLICY_COLORS.items():
        if policy == "Belady's" or policy not in results:
            continue

        gaps = []
        for w in WORKLOADS:
            belady_hr = results["Belady's"].get(w, {}).get('hit_rate', 0)
            policy_hr = results[policy].get(w, {}).get('hit_rate', 0)
            gap       = (belady_hr - policy_hr) * 100
            gaps.append(round(gap, 2))

        fig.add_trace(go.Scatter(
            name  = policy,
            x     = WORKLOAD_LABELS,
            y     = gaps,
            mode  = 'lines+markers',
            line  = dict(color=color, width=2),
            marker= dict(size=10),
            text  = [f"{g:.1f}% gap" for g in gaps],
            hovertemplate = "%{text}<extra>%{fullData.name}</extra>",
        ))

    # Add Belady's as zero line (optimal)
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color=POLICY_COLORS["Belady's"],
        annotation_text="Belady's Optimal (0% gap)",
        annotation_position="right",
    )

    fig.update_layout(
        title={'text': "Gap to Belady's Optimal Hit Rate", 'font': {'size': 20}},
        yaxis_title = 'Gap to Optimal (%)',
        xaxis_title = 'Workload Pattern',
        legend      = dict(orientation='h', y=-0.15),
        height      = 500,
        template    = 'plotly_white',
    )

    path = os.path.join(cfg['plots_dir'], '03_gap_to_optimal.html')
    fig.write_html(path)
    print(f"Chart 3 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: Training Convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(monitor, lru_baseline=None, cfg=CONFIG):
    """
    Line chart: DQN hit rate per training episode.
    Shows the AI learning over time.
    Includes LRU baseline as reference.
    """
    _ensure_plots_dir(cfg)
    episodes = list(range(1, len(monitor.episode_hits) + 1))

    fig = go.Figure()

    # DQN hit rate per episode
    fig.add_trace(go.Scatter(
        name  = 'DQN Hit Rate',
        x     = episodes,
        y     = [h * 100 for h in monitor.episode_hits],
        mode  = 'lines+markers',
        line  = dict(color=POLICY_COLORS['DQN'], width=3),
        marker= dict(size=8),
    ))

    # LRU baseline
    if lru_baseline is not None:
        fig.add_hline(
            y               = lru_baseline * 100,
            line_dash       = 'dash',
            line_color      = POLICY_COLORS['LRU'],
            annotation_text = f"LRU Baseline ({lru_baseline*100:.1f}%)",
            annotation_position = "right",
        )

    # Switch point (if hybrid mode)
    if monitor.switch_episode is not None:
        fig.add_vline(
            x               = monitor.switch_episode,
            line_dash       = 'dot',
            line_color      = 'purple',
            annotation_text = 'Switched to DQN',
        )

    # Epsilon decay on secondary axis
    if monitor.epsilon_history:
        fig.add_trace(go.Scatter(
            name   = 'Epsilon',
            x      = episodes,
            y      = monitor.epsilon_history,
            mode   = 'lines',
            line   = dict(color='orange', width=1, dash='dot'),
            yaxis  = 'y2',
            opacity = 0.7,
        ))

    fig.update_layout(
        title={'text': 'DQN Training Convergence', 'font': {'size': 20}},
        xaxis_title = 'Training Episode',
        yaxis       = dict(title='Hit Rate (%)', side='left'),
        yaxis2      = dict(
            title      = 'Epsilon',
            side       = 'right',
            overlaying = 'y',
            range      = [0, 1],
        ),
        legend  = dict(orientation='h', y=-0.15),
        height  = 500,
        template = 'plotly_white',
    )

    path = os.path.join(cfg['plots_dir'], '04_convergence.html')
    fig.write_html(path)
    print(f"Chart 4 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: Writeback Rate
# ─────────────────────────────────────────────────────────────────────────────

def plot_writeback_rates(results, cfg=CONFIG):
    """
    Grouped bar chart: writeback rate for LRU, LFU, DQN.
    Shows DQN learned to avoid dirty evictions (your unique contribution).
    Lower = better.
    """
    _ensure_plots_dir(cfg)
    policies_to_show = ['LRU', 'LFU', 'DQN']

    fig = go.Figure()

    for policy in policies_to_show:
        if policy not in results:
            continue
        color  = POLICY_COLORS[policy]
        y_vals = [
            results[policy].get(w, {}).get('writeback_rate', 0) * 100
            for w in WORKLOADS
        ]
        fig.add_trace(go.Bar(
            name         = policy,
            x            = WORKLOAD_LABELS,
            y            = y_vals,
            marker_color = color,
            text         = [f"{v:.1f}%" for v in y_vals],
            textposition = 'outside',
        ))

    fig.add_annotation(
        text="Lower writeback rate = fewer expensive write-backs to RAM",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=11, color='gray')
    )

    fig.update_layout(
        title={'text': 'Dirty Block Writeback Rate by Policy',
               'font': {'size': 20}},
        barmode     = 'group',
        yaxis_title = 'Writeback Rate (%)',
        xaxis_title = 'Workload Pattern',
        legend      = dict(orientation='h', y=-0.15),
        height      = 500,
        template    = 'plotly_white',
    )

    path = os.path.join(cfg['plots_dir'], '05_writeback_rate.html')
    fig.write_html(path)
    print(f"Chart 5 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6: Training Health Monitor
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_health(monitor, cfg=CONFIG):
    """
    Dual-axis chart: epsilon decay + safety overrides per episode.
    Shows training was healthy.

    Healthy training:
    - Epsilon drops from 1.0 to 0.05 smoothly
    - Safety overrides decrease over episodes (DQN learning)
    """
    _ensure_plots_dir(cfg)
    episodes = list(range(1, len(monitor.epsilon_history) + 1))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Safety overrides (bar)
    fig.add_trace(go.Bar(
        name         = 'Safety Overrides',
        x            = episodes,
        y            = monitor.safety_overrides,
        marker_color = 'rgba(231,76,60,0.4)',
    ), secondary_y=True)

    # Epsilon decay (line)
    fig.add_trace(go.Scatter(
        name = 'Epsilon',
        x    = episodes,
        y    = monitor.epsilon_history,
        mode = 'lines+markers',
        line = dict(color='orange', width=2),
    ), secondary_y=False)

    # Loss (line)
    if monitor.loss_history:
        fig.add_trace(go.Scatter(
            name = 'Avg Loss',
            x    = episodes,
            y    = monitor.loss_history,
            mode = 'lines',
            line = dict(color='purple', width=2, dash='dot'),
        ), secondary_y=False)

    fig.update_layout(
        title={'text': 'Training Health Monitor', 'font': {'size': 20}},
        xaxis_title = 'Training Episode',
        legend      = dict(orientation='h', y=-0.15),
        height      = 500,
        template    = 'plotly_white',
    )
    fig.update_yaxes(title_text="Epsilon / Loss", secondary_y=False)
    fig.update_yaxes(title_text="Safety Overrides", secondary_y=True)

    path = os.path.join(cfg['plots_dir'], '06_training_health.html')
    fig.write_html(path)
    print(f"Chart 6 saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MASTER DASHBOARD — All 6 charts in one file
# ─────────────────────────────────────────────────────────────────────────────

def generate_master_dashboard(results, monitor=None, cfg=CONFIG):
    """
    Generate all 6 charts and save each individually.
    Also prints a summary table to console.

    Args:
        results: nested dict from evaluation mode
        monitor: TrainingMonitor from training mode (optional)
        cfg:     config dict
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION DASHBOARD")
    print("="*60 + "\n")

    _ensure_plots_dir(cfg)

    # Generate all charts
    plot_hit_rate_comparison(results, cfg)
    plot_amat_comparison(results, cfg)
    plot_gap_to_optimal(results, cfg)
    plot_writeback_rates(results, cfg)

    if monitor is not None:
        plot_convergence(monitor, cfg=cfg)
        plot_training_health(monitor, cfg)

    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY TABLE")
    print("="*60)
    print(f"\n{'Policy':12} | {'Pattern':10} | {'HitRate':8} | "
          f"{'AMAT':8} | {'WritebackRate':13}")
    print("-" * 60)

    for policy in ['FIFO', 'LRU', 'LFU', 'DQN', "Belady's"]:
        if policy not in results:
            continue
        for pattern in WORKLOADS:
            if pattern not in results[policy]:
                continue
            m = results[policy][pattern]
            print(f"{policy:12} | {pattern:10} | "
                  f"{m.get('hit_rate',0)*100:6.2f}% | "
                  f"{m.get('amat',0):7.1f}  | "
                  f"{m.get('writeback_rate',0)*100:6.2f}%")

    print("\n" + "="*60)
    print(f"All plots saved to: {cfg['plots_dir']}")
    print("Open any .html file in a browser to view interactive charts.")
    print("="*60)
```

---

## 6.22 Main Entry Point

**File**: `main.py`

```python
# main.py
# =============================================================================
# MAIN ENTRY POINT
# Selects operating mode and runs corresponding pipeline
# =============================================================================

import sys
import os
from config import CONFIG, validate_config


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1].lower()
    validate_config(CONFIG)

    if mode == 'train':
        from training_mode import run_training_mode
        monitor, agent = run_training_mode(CONFIG)

    elif mode == 'eval':
        if not os.path.exists(CONFIG['model_path']):
            print(f"❌ Model not found: {CONFIG['model_path']}")
            print("   Run training first: python main.py train")
            sys.exit(1)
        from evaluation_mode import run_evaluation_mode
        results = run_evaluation_mode(CONFIG)

    elif mode == 'visualize':
        if not os.path.exists(CONFIG['results_path']):
            print(f"❌ Results not found: {CONFIG['results_path']}")
            print("   Run evaluation first: python main.py eval")
            sys.exit(1)
        from evaluation_mode import load_results
        from visualization.dashboard import generate_master_dashboard
        results = load_results(CONFIG['results_path'])
        generate_master_dashboard(results, monitor=None, cfg=CONFIG)

    elif mode == 'hybrid':
        if not os.path.exists(CONFIG['model_path']):
            print(f"❌ Model not found: {CONFIG['model_path']}")
            print("   Run training first: python main.py train")
            sys.exit(1)
        from hybrid_mode import run_hybrid_mode
        run_hybrid_mode(CONFIG)

    elif mode == 'validate':
        # Quick validation — run decoder, config, trace generator checks
        from cache.decoder import AddressDecoder
        from data.generator import generate_trace
        from data.validator import validate_trace, validate_zipfian

        decoder = AddressDecoder(CONFIG)
        decoder.validate()

        trace = generate_trace('zipfian', seed=42)
        validate_trace(trace, 'zipfian')
        validate_zipfian(trace)
        print("\nAll validation checks passed ✅")

    elif mode == 'test':
        # Run test suite
        import unittest
        loader = unittest.TestLoader()
        suite  = loader.discover('tests')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    else:
        print(f"❌ Unknown mode: {mode}")
        print_usage()
        sys.exit(1)


def print_usage():
    print("""
Adaptive Cache Simulator — Usage:

  python main.py train      → Train DQN agent offline
  python main.py eval       → Evaluate all policies, save results
  python main.py visualize  → Generate interactive dashboard
  python main.py hybrid     → Live LRU→DQN transition demo
  python main.py validate   → Validate config and generators
  python main.py test       → Run full test suite

Recommended order:
  1. python main.py validate   (verify setup)
  2. python main.py test       (verify correctness)
  3. python main.py train      (train the DQN)
  4. python main.py eval       (benchmark all policies)
  5. python main.py visualize  (generate charts)
  6. python main.py hybrid     (demo the transition)
    """)


if __name__ == '__main__':
    main()
```

---

# 7. COMPLETE TEST SUITE

**File**: `tests/test_decoder.py`

```python
import unittest
import sys
sys.path.insert(0, '..')
from config import CONFIG
from cache.decoder import AddressDecoder


class TestAddressDecoder(unittest.TestCase):

    def setUp(self):
        self.decoder = AddressDecoder(CONFIG)

    def test_decode_reconstruct_zero(self):
        tag, idx, off = self.decoder.decode(0x00000000)
        self.assertEqual(self.decoder.reconstruct(tag, idx, off), 0x00000000)

    def test_decode_reconstruct_max(self):
        addr = (1 << CONFIG['address_bits']) - 1
        tag, idx, off = self.decoder.decode(addr)
        self.assertEqual(self.decoder.reconstruct(tag, idx, off), addr)

    def test_decode_reconstruct_typical(self):
        for addr in [0x00001A3F, 0x00002B10, 0x0000FFFF, 12345]:
            tag, idx, off = self.decoder.decode(addr)
            reconstructed = self.decoder.reconstruct(tag, idx, off)
            self.assertEqual(reconstructed, addr, f"Failed for {addr:#010x}")

    def test_offset_within_block_size(self):
        for addr in range(0, 10000, 64):
            _, _, offset = self.decoder.decode(addr)
            self.assertLess(offset, CONFIG['block_size'])

    def test_index_within_num_sets(self):
        num_sets = max(CONFIG['num_sets'], 1)
        for addr in range(1000):
            _, index, _ = self.decoder.decode(addr)
            self.assertLess(index, num_sets)

    def test_validate_passes(self):
        self.assertTrue(self.decoder.validate())
```

**File**: `tests/test_block.py`

```python
import unittest
import sys
sys.path.insert(0, '..')
from cache.block import CacheBlock


class TestCacheBlock(unittest.TestCase):

    def test_initial_state_clean(self):
        block = CacheBlock(tag=1)
        self.assertFalse(block.dirty)

    def test_write_sets_dirty(self):
        block = CacheBlock(tag=1)
        block.write()
        self.assertTrue(block.dirty)

    def test_read_does_not_set_dirty(self):
        block = CacheBlock(tag=1)
        block.read()
        self.assertFalse(block.dirty)

    def test_frequency_increments_on_read(self):
        block = CacheBlock(tag=1)
        initial = block.frequency
        block.read()
        self.assertEqual(block.frequency, initial + 1)

    def test_frequency_increments_on_write(self):
        block = CacheBlock(tag=1)
        initial = block.frequency
        block.write()
        self.assertEqual(block.frequency, initial + 1)

    def test_recency_resets_on_read(self):
        block = CacheBlock(tag=1)
        block.tick(); block.tick(); block.tick()
        self.assertEqual(block.recency, 3)
        block.read()
        self.assertEqual(block.recency, 0)

    def test_recency_resets_on_write(self):
        block = CacheBlock(tag=1)
        block.tick(); block.tick()
        block.write()
        self.assertEqual(block.recency, 0)

    def test_tick_increments_age_and_recency(self):
        block = CacheBlock(tag=1)
        block.tick()
        self.assertEqual(block.age, 1)
        self.assertEqual(block.recency, 1)

    def test_to_state_normalized(self):
        block = CacheBlock(tag=1)
        state = block.to_state(max_val=1000)
        self.assertEqual(len(state), 4)
        for val in state:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

    def test_dirty_in_state(self):
        block = CacheBlock(tag=1)
        self.assertEqual(block.to_state()[2], 0.0)  # clean
        block.write()
        self.assertEqual(block.to_state()[2], 1.0)  # dirty
```

**File**: `tests/test_buffer.py`

```python
import unittest
import sys
import numpy as np
sys.path.insert(0, '..')
from ml.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def test_empty_not_ready(self):
        buf = ReplayBuffer(capacity=100, min_size=10)
        self.assertFalse(buf.ready())

    def test_ready_after_min_size(self):
        buf = ReplayBuffer(capacity=100, min_size=10)
        for _ in range(10):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        self.assertTrue(buf.ready())

    def test_capacity_not_exceeded(self):
        buf = ReplayBuffer(capacity=50, min_size=10)
        for _ in range(100):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        self.assertEqual(len(buf), 50)

    def test_sample_correct_size(self):
        buf = ReplayBuffer(capacity=100, min_size=10)
        for _ in range(50):
            buf.push(np.zeros(16), 0, 1.0, np.zeros(16))
        states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
        self.assertEqual(len(states), 8)
        self.assertEqual(len(actions), 8)
```

**File**: `tests/test_policies.py`

```python
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
        Step 1: A → MISS [A]
        Step 2: B → MISS [A,B] (full)
        Step 3: C → MISS evict LRU=A → [B,C]
        Step 4: A → MISS evict LRU=B → [C,A]
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
        Trace A B A B A B → 4 hits, 2 misses for 2-way cache
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
```

---

# 8. RUNNING THE PROJECT

## Recommended Execution Order

```bash
# Step 1: Verify setup
python main.py validate

# Step 2: Run test suite (all must pass before proceeding)
python main.py test

# Step 3: Train DQN (takes 5-30 minutes depending on hardware)
python main.py train

# Step 4: Evaluate all policies
python main.py eval

# Step 5: Generate charts
python main.py visualize

# Step 6: Demo hybrid transition
python main.py hybrid
```

## Expected Outputs

After `python main.py train`:
```
models/dqn_cache.pth    ← saved model
```

After `python main.py eval`:
```
results/results.csv     ← all benchmark results
```

After `python main.py visualize`:
```
plots/01_hit_rate.html
plots/02_amat.html
plots/03_gap_to_optimal.html
plots/04_convergence.html
plots/05_writeback_rate.html
plots/06_training_health.html
```

Open any `.html` file in Chrome/Firefox for interactive charts.

---

# 9. WHAT RESULTS TO EXPECT

## Realistic Hit Rate Ranges (for our config)

These are approximate — your actual results will vary:

| Policy | Sequential | Random | Stride | Zipfian |
|---|---|---|---|---|
| FIFO | 50-65% | 5-15% | 40-60% | 40-60% |
| LRU | 55-70% | 5-15% | 40-65% | 50-70% |
| LFU | 50-65% | 5-15% | 35-55% | 45-65% |
| DQN | 55-75% | 5-15% | 40-70% | 55-75% |
| Belady's | 65-85% | 15-25% | 55-80% | 65-85% |

## What "Success" Looks Like

DQN wins if it beats LRU on 3 out of 4 workloads by AMAT.
DQN on Random will likely match or lose to LRU — that is expected and fine.

## What To Do If DQN Underperforms

Go to Section 10 — Debugging Guide.

---

# 10. DEBUGGING GUIDE

If DQN doesn't beat LRU, follow this checklist in order:

## Debug Checklist

### Check 1: Input Normalization
```python
# Add to training loop temporarily
state = cache_set.get_state()
print(f"State min={state.min():.3f} max={state.max():.3f}")
# All values should be between 0 and 1
# If values > 1: normalization broken
```

### Check 2: Reward Values
```python
# Add reward logging temporarily
print(f"Reward: {reward:.2f}")
# If all rewards are -1.0: agent never causes hits (check miss logic)
# If all rewards are 0: reward function not being called
```

### Check 3: Epsilon Decay
```python
# Print epsilon every 100 steps
if step % 100 == 0:
    print(f"Step {step}: epsilon={agent.epsilon:.3f}")
# Should decrease from 1.0 toward 0.05 over training
```

### Check 4: Replay Buffer Growth
```python
# Print buffer size periodically
if step % 100 == 0:
    print(f"Buffer: {len(agent.replay_buffer)}")
# Should grow until capacity, then stay at 10000
```

### Check 5: Loss Trend
```python
# Track loss per episode
# Loss should generally decrease over episodes
# NaN loss → learning rate too high, reduce to 0.0001
# Loss not decreasing → target sync broken
```

### Check 6: Reduce Problem Size
```python
# Temporarily change config
CONFIG['associativity'] = 2   # only 2 choices, easier to learn
CONFIG['trace_length']  = 1000
# If DQN can't beat LRU with 2-way cache → fundamental bug
```

### Check 7: Safety Guard Rate
```python
# High safety overrides throughout training → agent not learning
print(f"Safety overrides this episode: {overrides}")
# Should decrease over episodes
# If always high → reward shaping not working
```

---

# 11. VIVA PREPARATION

## Top 10 Questions You Will Be Asked

### Q1: "Why DQN and not supervised learning?"
> "Supervised learning requires labeled data — you'd need to know in advance which eviction decision was optimal. DQN learns from consequences: it makes a decision, observes whether the next access was a hit or miss, and updates accordingly. This is more natural for cache replacement where the ground truth is only revealed after the fact."

### Q2: "How is this different from LeCaR?"
> "LeCaR uses regret minimization to switch between LRU and LFU — it never directly decides which block to evict. My DQN operates at the block level, making direct eviction decisions. Additionally, LeCaR targets storage caches and ignores write semantics. My reward function explicitly penalizes dirty block evictions with a -0.2 penalty, making it more appropriate for CPU cache simulation."

### Q3: "Why can't DQN beat Belady's?"
> "Belady's is theoretically optimal because it uses future knowledge — it knows exactly when each block will be accessed next. DQN only has access to past access history (recency, frequency, age). No online algorithm can beat Belady's — this is proven by Belady's theorem. So Belady's serves as our theoretical ceiling, not a target to beat."

### Q4: "What is your state space?"
> "For each block in the cache set, I extract 4 features: recency (how long since last access), frequency (total access count), dirty bit (0 or 1), and age (time since loaded). With 4-way associativity this gives a 16-dimensional state vector. All features are normalized to [0,1] by dividing by the trace length."

### Q5: "What is your reward function?"
> "+1 for cache hit, -1 for cache miss, -0.2 for evicting a dirty block, and a recency bonus of up to +0.3 for evicting stale blocks. The dirty penalty is intentionally smaller than the miss penalty because a writeback is costly but less harmful than a cache miss."

### Q6: "Why write-back policy and not write-through?"
> "Write-back is used in real L2 caches because it reduces RAM bus traffic — we only write to RAM when a dirty block is evicted, not on every write. Write-through would eliminate the dirty bit problem but is unrealistic and reduces the ML challenge since all blocks would always be clean."

### Q7: "What is AMAT and why is it a better metric than hit rate?"
> "Average Memory Access Time = hit_time + (miss_rate × miss_penalty) + (writeback_rate × writeback_cycles). It's better than hit rate alone because it accounts for the actual time cost of misses and writebacks. A policy with slightly lower hit rate but much lower writeback rate might have better AMAT — which is what matters for real performance."

### Q8: "How did you handle the cold start problem?"
> "I implemented a confidence-based warm-up: the system starts with LRU and runs DQN in shadow mode. The DQN agent's confidence is measured as the gap between the best and second-best Q-values. When confidence exceeds 0.3 and the replay buffer has 500+ experiences, the system switches to DQN. For clean benchmarking, I pre-train the agent offline and evaluate without any warm-up phase."

### Q9: "Could this be implemented in real hardware?"
> "In principle yes, but the ML inference would need to be implemented as dedicated logic — likely an FPGA or ASIC accelerator. This introduces area and power overhead that needs to be justified by the performance gain. That's why this project focuses on simulation first — validating the policy concept before considering hardware implementation. This is standard practice in computer architecture research."

### Q10: "Why is your cache only 256 bytes? That's tiny."
> "Deliberately small. A larger cache has fewer evictions — which means fewer decisions for the DQN to learn from and fewer opportunities to demonstrate improvement over LRU. Smaller cache = higher miss rate = replacement policy matters more. This is consistent with LeCaR's findings that ML-based policies show the largest improvement when cache size is small relative to the working set."

---

# 12. RELATED WORK

## Papers You Must Know

### 1. LeCaR (2018) — Most Similar to Your Work
- **Full title**: "Driving Cache Replacement with ML-based LeCaR"
- **Authors**: Vietri et al.
- **Venue**: USENIX HotStorage 2018
- **What it does**: Uses regret minimization RL to switch between LRU and LFU
- **Key result**: Outperforms ARC by 18x on small caches
- **Read it**: https://www.usenix.org/conference/hotstorage18/presentation/vietri
- **How yours differs**: Block-level DQN decisions vs policy switching; dirty block handling; confidence-based cold start

### 2. Hawkeye (2016) — Most Cited
- **Full title**: "Hawkeye: A Cache Replacement Policy Inspired by Optimal"
- **Authors**: Jain & Lin
- **Venue**: ISCA 2016 (won Cache Replacement Championship)
- **What it does**: Uses Belady's as training signal for a binary classifier
- **How yours differs**: RL not supervised; software simulator not hardware; visualization

### 3. Mockingjay (2022) — State of the Art
- **Full title**: "Mockingjay: Seed Sampling and Robust Reuse Distance Prediction"
- **Authors**: Shah et al.
- **Venue**: ISCA 2022
- **What it does**: Predicts reuse distance with high accuracy
- **How yours differs**: RL approach; accessible implementation; different target audience

## Your Novelty Statement
> "This project implements a DQN-based L2 cache replacement policy with three differentiators from prior work: (1) explicit dirty block penalty in the reward function, addressing write-back overhead not modeled in LeCaR; (2) a confidence-based warm-up policy solving the cold start problem; (3) an interactive visualization dashboard comparing all policies against Belady's theoretical optimum."

---

# 13. FUTURE WORK — V2 WITH L1

**DO NOT start this until v1 is 100% complete and working.**

## Prerequisites for V2
```
□ All v1 tests passing
□ DQN training converging
□ Full evaluation results collected and analyzed
□ Everything committed to git
□ Create new branch: git checkout -b v2-multilevel
```

## V2 Architecture
```
[CPU] → [L1 Cache (32B, 2-way, LRU)] → [L2 Cache (256B, 4-way, DQN)] → [RAM]
```

## New Design Decisions for V2
- **Inclusion policy**: Exclusive (L1 evictions move to L2, not discarded)
- **AI policy**: DQN at L2 only (L1 uses LRU — near optimal at small scale)
- **AMAT formula**: Recursive — `AMAT_L1 = L1_hit + L1_miss_rate × AMAT_L2`
- **New tests**: All 5 current test files need extensions for L1 behavior

---

# 14. COMPLETE CONFIG REFERENCE

All parameters in one place with explanations:

| Parameter | Value | Why |
|---|---|---|
| cache_level | 'L2' | We simulate L2 |
| cache_size | 256 | Small for more evictions |
| block_size | 64 | Industry standard |
| associativity | 4 | Standard L2 |
| address_bits | 32 | Standard address space |
| hit_cycles | 12 | L2 hit latency |
| miss_cycles | 100 | DRAM latency |
| writeback_cycles | 100 | DRAM write latency |
| trace_length | 10000 | Sufficient for training |
| write_ratio | 0.3 | 30% writes — realistic |
| zipfian_alpha | 1.2 | Strong realistic skew |
| address_space | 1024 | Distinct address pool |
| stride_step | 8 | Common matrix stride |
| train_test_split | 0.7 | 70/30 standard |
| random_seeds | [42,123,777] | Fixed for reproducibility |
| replay_buffer_size | 10000 | Sweet spot |
| batch_size | 32 | Standard DQN |
| min_buffer_size | 500 | Before training starts |
| epsilon_start | 1.0 | Fully random at start |
| epsilon_end | 0.05 | 5% random at end |
| epsilon_decay | 0.995 | Gradual decay |
| hidden_size | 64 | Sufficient for problem |
| learning_rate | 0.001 | Adam default |
| gamma | 0.99 | High future discount |
| target_sync_steps | 100 | Stable training |
| training_episodes | 10 | Enough to converge |
| train_frequency | 20 | Train every 20 steps |
| hit_reward | 1.0 | Primary signal |
| miss_penalty | -1.0 | Primary negative |
| dirty_penalty | -0.2 | Smaller than miss |
| recency_bonus_max | 0.3 | Hint for agent |
| switch_threshold | 0.05 | 5% better to switch |
| switch_window | 100 | Evaluate over 100 steps |
| confidence_threshold | 0.3 | Q-value gap |
| early_stop_patience | 3 | Episodes without improvement |
| early_stop_min_improve | 0.01 | Minimum 1% improvement |

---

*End of Master Document*

---

> **REMEMBER**
> Build in this order:
> 1. config.py → decoder → block → ram → simulator
> 2. FIFO → LRU → LFU → Belady's
> 3. generator → validator
> 4. network → replay_buffer → monitor → agent → dqn policy
> 5. training_mode → evaluation_mode → hybrid_mode
> 6. dashboard
> 7. main.py
> 8. tests
>
> Test after every step. Never add complexity on top of broken code.
