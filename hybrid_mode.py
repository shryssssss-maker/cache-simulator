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
    print("HYBRID MODE — Live LRU->DQN Transition")
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
                  f"eps={agent.epsilon:.3f}")

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
