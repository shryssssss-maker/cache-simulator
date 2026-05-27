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
    print("\n[1/6] Generating Mixed training trace (50% Zipf, 25% Random, 25% Stride)...")
    n_total = cfg['trace_length']
    n_zipf = int(n_total * 0.5)
    n_rand = int(n_total * 0.25)
    n_stride = n_total - n_zipf - n_rand
    
    trace_zipf = generate_trace('zipfian', seed=cfg['random_seeds'][0], n=n_zipf)
    trace_rand = generate_trace('random', seed=cfg['random_seeds'][1], n=n_rand)
    trace_stride = generate_trace('stride', seed=cfg['random_seeds'][2], n=n_stride)
    
    trace = trace_zipf + trace_rand + trace_stride
    validate_trace(trace, 'mixed')

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
    print(f"  Table size: {sum(len(indices) for indices in belady_policy._occurrences.values())} total occurrences recorded")

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

            # Epsilon decay moved to per-episode

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

        # Decay epsilon at the end of the episode
        agent.decay_epsilon()

        # Check early stopping
        if monitor.should_early_stop():
            break

        # Learning status
        learning = monitor.is_learning()
        if learning is False and episode > 3:
            print(f"[WARNING] hit rate declining - check debug guide")

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
