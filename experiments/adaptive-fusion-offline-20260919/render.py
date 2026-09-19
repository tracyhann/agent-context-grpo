#!/usr/bin/env python3
"""Render the completed offline adaptive-fusion audit as exportable figures."""
from pathlib import Path
import csv
import json
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = [json.loads(p.read_text()) for p in sorted((HERE / 'audit/per_step').glob('*.json'))]
summary = json.loads((HERE / 'audit/summary.json').read_text())
fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
colors = {'original': '#555555', 'row': '#bc4b2d', 'task': '#246b91', 'terminal': '#8e6caa'}
for row, tag in enumerate(['M10', 'M11']):
    records = sorted([x for x in rows if x['tag'] == tag], key=lambda x: x['step'])
    steps = np.array([x['step'] for x in records])
    def curve(ax, variant, field, label, color):
        values = np.array([x['variants'][variant].get(field, np.nan) for x in records], dtype=float)
        smooth = np.array([np.mean(values[max(0, i - 4):i + 1]) for i in range(len(values))])
        ax.plot(steps, smooth, label=label, color=color, linewidth=1.8)
    curve(axes[row, 0], 'row_snr', 'weight_nonterminal_mean', 'Nonterminal weight', colors['task'])
    curve(axes[row, 0], 'row_snr', 'weight_terminal_mean', 'Terminal weight', colors['terminal'])
    axes[row, 0].set_ylim(0, 1); axes[row, 0].set_title(f'{tag}: row-gate weights')
    curve(axes[row, 1], 'fixed_1', 'terminal_mass_share', 'Original / uniform scaling', colors['original'])
    curve(axes[row, 1], 'row_snr', 'terminal_mass_share', 'Row SNR gate', colors['row'])
    curve(axes[row, 1], 'task_nonterminal_snr', 'terminal_mass_share', 'Task gate from nonterminal SNR', colors['task'])
    axes[row, 1].set_ylim(0, 1); axes[row, 1].set_title(f'{tag}: terminal share of |future credit|')
    curve(axes[row, 2], 'row_snr', 'task_mean_abs', 'Row SNR gate', colors['row'])
    curve(axes[row, 2], 'row_snr_centered', 'task_mean_abs', 'Row gate, re-centered', colors['terminal'])
    curve(axes[row, 2], 'task_nonterminal_snr', 'task_mean_abs', 'Task gate', colors['task'])
    axes[row, 2].set_title(f'{tag}: mean |task-mean future credit|')
    for ax in axes[row]:
        ax.grid(alpha=.2); ax.set_xlabel('Optimizer step'); ax.legend(fontsize=8)
fig.suptitle('Offline adaptive fusion: conditional bootstrap sensitivity and credit redistribution\nFive-batch moving means; these are not evaluation success rates.', fontsize=13)
fig.savefig(HERE / 'adaptive_fusion.png', dpi=160)
fig.savefig(HERE / 'adaptive_fusion.pdf')
plt.close(fig)

with (HERE / 'comparison.csv').open('w') as f:
    fields = ['run', 'period', 'variant', 'weight_mean', 'weight_nonterminal_mean', 'weight_terminal_mean',
              'future_mass_retention', 'terminal_mass_share', 'nonterminal_mass_retention', 'task_mean_abs',
              'frozen_norm_noise_fraction', 'lone_nonterminal_rows', 'lone_nonterminal_future_negative_rows',
              'lone_nonterminal_combined_negative_rows', 'history_sign_flip_fraction']
    writer = csv.DictWriter(f, fields); writer.writeheader()
    for run, periods in summary['results'].items():
        for period, stats in periods.items():
            for variant, values in stats['variants'].items():
                writer.writerow(dict(run=run, period=period, variant=variant, **{k: values.get(k) for k in fields[3:]}))
print(json.dumps({'rows': len(rows), 'figure': str(HERE / 'adaptive_fusion.png'), 'table': str(HERE / 'comparison.csv')}))
