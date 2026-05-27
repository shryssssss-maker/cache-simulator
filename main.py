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
        print("\nAll validation checks passed [OK]")

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
