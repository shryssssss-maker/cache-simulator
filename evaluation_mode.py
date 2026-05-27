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
                    policy_name, agent, test_trace, decoder, cfg
                )

                # Run simulation
                metrics = run_simulation(policy, test_trace, cfg)
                run_results.append(metrics)

            # Average over seeds
            avg = _average_results(run_results)

            if policy_name not in results:
                results[policy_name] = {}
            results[policy_name][pattern] = avg

            print(f"    -> HitRate={avg['hit_rate']:.3f} | "
                  f"AMAT={avg['amat']:.1f} | "
                  f"WritebackRate={avg['writeback_rate']:.3f}")

    # Save results
    _save_results(results, cfg['results_path'])

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"  Results saved: {cfg['results_path']}")
    print("="*60)

    return results


def _build_policy(name, agent, test_trace, decoder, cfg):
    """Build a fresh policy instance for one run."""

    if name == 'FIFO':
        return FIFOPolicy()

    elif name == 'LRU':
        return LRUPolicy()

    elif name == 'LFU':
        return LFUPolicy()

    elif name == "Belady's":
        # Precompute ONLY on the test trace so step indices align (0 to len-1)
        belady = BeladyPolicy()
        tag_trace = extract_tags(test_trace, decoder)
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

    print(f"\nResults saved -> {path}")


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
