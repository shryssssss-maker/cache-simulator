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
    'DQN'      : '#2ECC71',   # green  <- your AI
    "Belady's" : '#3498DB',   # blue   <- optimal ceiling
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
    print(f"Chart 1 saved -> {path}")
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
    print(f"Chart 2 saved -> {path}")
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
        print("Warning: Belady's results not found - skipping gap chart")
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
    print(f"Chart 3 saved -> {path}")
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
    print(f"Chart 4 saved -> {path}")
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
    print(f"Chart 5 saved -> {path}")
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
    print(f"Chart 6 saved -> {path}")
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
