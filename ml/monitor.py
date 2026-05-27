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
              f"eps={epsilon:.3f} | "
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
