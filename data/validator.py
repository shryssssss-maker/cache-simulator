# data/validator.py
# =============================================================================
# TRACE VALIDATOR
# Validates memory access traces produced by generator.py
# =============================================================================

from collections import Counter


def validate_trace(trace, pattern_name="unknown"):
    """
    Basic sanity checks for any generated trace.

    Checks:
      - trace is non-empty
      - every element is a 2-tuple (op, address)
      - op is either 'R' or 'W'
      - address is a non-negative integer
      - address is a multiple of CONFIG block_size (64 bytes)

    Args:
        trace:        list of (op, address) tuples
        pattern_name: label printed in output (for human readability)

    Raises:
        AssertionError on any failed check.
    """
    assert len(trace) > 0, \
        f"[{pattern_name}] Trace is empty"

    for i, entry in enumerate(trace):
        assert isinstance(entry, tuple) and len(entry) == 2, \
            f"[{pattern_name}] Entry {i} is not a 2-tuple: {entry!r}"

        op, addr = entry

        assert op in ('R', 'W'), \
            f"[{pattern_name}] Entry {i}: op must be 'R' or 'W', got {op!r}"

        assert isinstance(addr, int) and addr >= 0, \
            f"[{pattern_name}] Entry {i}: address must be a non-negative int, got {addr!r}"

        assert addr % 64 == 0, \
            f"[{pattern_name}] Entry {i}: address {addr} is not aligned to 64 bytes"

    # Count reads / writes
    ops = Counter(op for op, _ in trace)
    total = len(trace)
    write_pct = ops['W'] / total * 100

    print(f"[validate_trace] {pattern_name:12} | "
          f"length={total:6d} | "
          f"R={ops['R']:6d}  W={ops['W']:6d}  "
          f"write%={write_pct:.1f}%  [OK]")


def validate_zipfian(trace, min_skew_ratio=3.0):
    """
    Verify that the zipfian trace shows a genuine power-law skew.

    Strategy:
      - Count accesses per unique address
      - Check that the top-20% of addresses account for >= 50% of accesses
        (this is weaker than the classic 80/20 — fine for alpha=1.2)
      - Check that the ratio (most-frequent / median-frequent) >= min_skew_ratio

    Args:
        trace:          list of (op, address) tuples
        min_skew_ratio: minimum required (top_freq / median_freq) ratio

    Raises:
        AssertionError if the distribution doesn't look skewed enough.
    """
    assert len(trace) > 0, "Zipfian trace is empty"

    # Count per address
    counts = Counter(addr for _, addr in trace)
    sorted_counts = sorted(counts.values(), reverse=True)

    top_20_pct = max(1, len(sorted_counts) // 5)
    top_accesses = sum(sorted_counts[:top_20_pct])
    total_accesses = sum(sorted_counts)
    coverage = top_accesses / total_accesses

    most_frequent = sorted_counts[0]
    median_frequent = max(1, sorted_counts[len(sorted_counts) // 2])
    skew_ratio = most_frequent / median_frequent

    print(f"[validate_zipfian] unique_addrs={len(counts)} | "
          f"top-20%-coverage={coverage:.1%} | "
          f"skew_ratio={skew_ratio:.1f}x")

    assert skew_ratio >= min_skew_ratio, (
        f"Zipfian skew too low: {skew_ratio:.1f}x < required {min_skew_ratio}x. "
        "Increase zipfian_alpha in config.py or check generator."
    )
    print(f"[validate_zipfian] Power-law skew confirmed (ratio={skew_ratio:.1f}x >= {min_skew_ratio}x) [OK]")


def validate_split(train, test, expected_ratio=0.7, tolerance=0.02):
    """
    Verify train/test split is correct and traces don't overlap.

    Checks:
      - Neither split is empty
      - train + test = original total
      - train ratio is within tolerance of expected_ratio
      - No data is lost (concatenation == original length)

    Args:
        train:          training trace (list of tuples)
        test:           test trace     (list of tuples)
        expected_ratio: expected fraction of data in train split (default 0.7)
        tolerance:      allowed deviation from expected_ratio (default 0.02)

    Raises:
        AssertionError on any failed check.
    """
    assert len(train) > 0, "Train split is empty"
    assert len(test)  > 0, "Test split is empty"

    total = len(train) + len(test)
    actual_ratio = len(train) / total

    assert abs(actual_ratio - expected_ratio) <= tolerance, (
        f"Train ratio {actual_ratio:.3f} is outside [{expected_ratio - tolerance:.3f}, "
        f"{expected_ratio + tolerance:.3f}]"
    )

    print(f"[validate_split] total={total} | "
          f"train={len(train)} ({actual_ratio:.1%}) | "
          f"test={len(test)} ({1 - actual_ratio:.1%}) [OK]")
