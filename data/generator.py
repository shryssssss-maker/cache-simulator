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