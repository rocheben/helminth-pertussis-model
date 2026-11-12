"""Figure A2 and summary of the global sensitivity analysis.
Usage: python figure_A2.py sensitivity_lhs.csv OUT_FOLDER"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.ticker
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1]); OUT = sys.argv[2]
df = df[df['ok'].astype(bool)].copy()
df['rel'] = (df['infants_s'] / df['infants_1'] - 1) * 100
P = ['tau', 'sigma', 'w50f', 'prev', 'kappa', 'R0', 'cov', 'eps', 'Tv', 'Tn', 'nu', 'acc']
LAB = {'tau': r'$\tau$ (infectiousness)', 'sigma': r'$\sigma$ (susceptibility)', 'w50f': 'half-saturation burden',
       'prev': 'helminth prevalence', 'kappa': r'aggregation $\kappa$', 'R0': r'pertussis $R_0$',
       'cov': 'vaccination coverage', 'eps': 'vaccine efficacy', 'Tv': 'protection after vaccination',
       'Tn': 'protection after infection', 'nu': 'boosting', 'acc': 'child-child contacts'}

def prcc(X, y):
    R = np.column_stack([rankdata(c) for c in X.T]); ry = rankdata(y); out = []
    for i in range(R.shape[1]):
        A = np.column_stack([np.delete(R, i, axis=1), np.ones(len(ry))])
        rx = R[:, i] - A @ np.linalg.lstsq(A, R[:, i], rcond=None)[0]
        rr = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        out.append(np.corrcoef(rx, rr)[0, 1])
    return np.array(out)
pc = prcc(df[P].values, df['rel'].values)
summ = dict(n=len(df), median=df.rel.median(), p90=df.rel.quantile(.9), max=df.rel.max(),
            gt10=(df.rel > 10).mean() * 100, gt25=(df.rel > 25).mean() * 100,
            Re1_min=df.Re1.min(), prcc={p: float(c) for p, c in zip(P, pc)})
for lo, hi in ((1, 1.25), (1.25, 2), (2, 3), (3, 5.01)):
    s = df[(df.tau >= lo) & (df.tau < hi)]
    summ[f'tau_{lo}_{hi}'] = dict(n=len(s), median=s.rel.median(), gt10=(s.rel > 10).mean() * 100,
                                   gt25=(s.rel > 25).mean() * 100, max=s.rel.max())
s = df[(df.tau < 1.25) & (df.rel > 10)]
summ['sus_only_gt10'] = s[['tau', 'sigma', 'w50f', 'R0', 'prev']].round(2).to_dict('records')
json.dump(summ, open(f'{OUT}/sensitivity_summary.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=lambda v: round(float(v), 2)))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
a = ax[0]
sc = a.scatter(df.tau, df.rel, c=df.R0, cmap='viridis_r', s=9, alpha=.8, lw=0)
a.set_xscale('log'); a.set_xticks([1, 2, 3, 5]); a.set_xticklabels(['1', '2', '3', '5']); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.axhline(10, color='0.6', ls=':', lw=1); a.axhline(25, color='0.6', ls=':', lw=1)
a.set_xlabel(r'$\tau$, maximal effect of helminths on infectiousness')
a.set_ylabel('Change in infant infections (%)')
a.set_title('A   Impact of the interaction', fontsize=9.5, loc='left')
cb = fig.colorbar(sc, ax=a, pad=.02); cb.set_label(r'pertussis $R_0$')
b = ax[1]
order = np.argsort(np.abs(pc))
b.barh([LAB[P[i]] for i in order], pc[order], color=['#c0392b' if pc[i] > 0 else '#1f5f8b' for i in order])
b.axvline(0, color='0.3', lw=.8); b.set_xlim(-1, 1)
b.set_xlabel('Partial rank correlation'); b.set_title('B   What drives it', fontsize=9.5, loc='left')
b.tick_params(axis='y', labelsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/FigureA2_sensitivity.png', dpi=300); fig.savefig(f'{OUT}/FigureA2_sensitivity.pdf')
