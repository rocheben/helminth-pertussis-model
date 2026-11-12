"""Figure A1 (Appendix A3). Requires hp_model.py. Usage: python figure_A1.py OUT_FOLDER"""
import sys, json, numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from hp_model import Model, BASE

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
CENTRAL = dict(BASE, sigma=10.0, tau=2.0)
WARM, COOL, GREY = '#c0392b', '#1f5f8b', '0.35'

def model(**kw): return Model(dict(CENTRAL, **kw)).calibrate()
def infants(M, sigma, tau, m=0.0):
    keep = M.P['tau']; M.P['tau'] = tau
    o = M.outcomes(M.equilibrium(sigma, m=m)); M.P['tau'] = keep; return o['infants']

M = model()
m20 = brentq(lambda m: 1 - M.helminth_children(M.lam_w, m)[0][0] - 0.2, 0, 200, xtol=1e-6)
MDA = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20])
prev = np.array([1 - M.helminth_children(M.lam_w, m)[0][0] for m in MDA]) * 100
age, stat, burden = M.masks(); sus = stat <= 1
def deltas(m, sigma):
    A0, A1, b, phi = M.build(sigma, m); x0 = spla.spsolve(A0.tocsc(), -b)
    w = (burden / (M.w50 + burden))
    d = 1 + (sigma - 1) * (x0 * w)[sus].sum() / x0[sus].sum()
    dp = 1 + (sigma - 1) * x0[sus & (burden > 0)].sum() / x0[sus].sum()
    return d, dp
D = {s: np.array([deltas(m, s)[0] for m in MDA]) for s in (2.0, 10.0, 100.0)}
Dprev = np.array([deltas(m, 10.0)[1] for m in MDA])

COV = np.linspace(0, 1, 11)
C = {k: [] for k in ('none', 't1', 't2', 't5', 'dew')}
for cv in COV:
    Mc = model(cov=float(cv))
    C['none'].append(infants(Mc, 1, 1)); C['t1'].append(infants(Mc, 10, 1))
    C['t2'].append(infants(Mc, 10, 2)); C['t5'].append(infants(Mc, 10, 5))
    C['dew'].append(infants(Mc, 10, 2, m20))
C = {k: np.array(v) for k, v in C.items()}

G = {}
for tau in (1.0, 2.0, 5.0):
    base = infants(M, 10, tau)
    G[tau] = np.array([(1 - infants(M, 10, tau, m) / base) * 100 for m in MDA])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.1))
xl = 'Worm clearance added by MDA (yr$^{-1}$)'
a = ax[0, 0]
a.plot(MDA, prev, color=COOL, lw=1.9); a.axhline(20, color='0.6', ls=':', lw=1)
a.text(MDA[-1], 21, 'WHO threshold for annual\npreventive chemotherapy', ha='right', va='bottom', fontsize=7.5, color='0.45')
a.set_ylim(0, 45); a.set_xlabel(xl); a.set_ylabel('Helminth prevalence, 1–14 yr (%)')
a.set_title('A   Helminth control works as intended', fontsize=9.5, loc='left')
b = ax[0, 1]
b.fill_between(MDA, D[2.0], D[100.0], color=WARM, alpha=.16, lw=0)
b.plot(MDA, D[10.0], color=WARM, lw=1.9); b.plot(MDA, Dprev, color=WARM, lw=1.1, ls=':')
b.axhline(1, color=GREY, ls='--', lw=1.3); b.set_yscale('log'); b.set_ylim(0.9, 25)
b.text(0.3, Dprev[0] * 1.1, 'from prevalence alone', fontsize=7.5, color=WARM)
b.set_xlabel(xl); b.set_ylabel(r'$\delta = S_r / S_f$')
b.set_title(r'B   ... and lowers $\delta$ ($\sigma$ from 2 to 100)', fontsize=9.5, loc='left')
c = ax[1, 0]
c.fill_between(COV * 100, C['t1'], C['t5'], color=WARM, alpha=.16, lw=0)
c.plot(COV * 100, C['t2'], color=WARM, lw=1.9)
c.plot(COV * 100, C['dew'], color=COOL, lw=1.3, ls='-.')
c.plot(COV * 100, C['none'], color=GREY, ls='--', lw=1.3)
c.set_ylim(0, None); c.set_xlabel('Pertussis vaccination coverage (%)'); c.set_ylabel('Infections per 1000 infants per year')
c.set_title(r'C   Infant infections ($\tau$ from 1 to 5)', fontsize=9.5, loc='left')
d = ax[1, 1]
d.fill_between(MDA, G[1.0], G[5.0], color=WARM, alpha=.16, lw=0)
d.plot(MDA, G[2.0], color=WARM, lw=1.9); d.axhline(0, color=GREY, ls='--', lw=1.3)
d.set_xlabel(xl); d.set_ylabel('Infant infections averted\nby deworming (%)')
d.set_title(r'D   Deworming benefit for infants ($\tau$ 1 to 5)', fontsize=9.5, loc='left')
handles = [Line2D([], [], color=GREY, ls='--', lw=1.3, label='no interaction'),
           Line2D([], [], color=WARM, lw=1.9, label=r'central case ($\sigma=10$, $\tau=2$)'),
           plt.Rectangle((0, 0), 1, 1, color=WARM, alpha=.16, label='range (see panel titles)'),
           Line2D([], [], color=COOL, lw=1.3, ls='-.', label='central case after deworming to 20% (C)')]
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.01), fontsize=8)
fig.tight_layout(rect=[0, .075, 1, 1])
fig.savefig(f'{OUT}/FigureA1_model.png', dpi=300); fig.savefig(f'{OUT}/FigureA1_model.pdf')
json.dump(dict(MDA=MDA.tolist(), prev=prev.tolist(), delta={str(k): v.tolist() for k, v in D.items()},
               delta_prev=Dprev.tolist(), COV=COV.tolist(), infants={k: v.tolist() for k, v in C.items()},
               averted={str(k): v.tolist() for k, v in G.items()}), open(f'{OUT}/FigureA1_values.json', 'w'), indent=1)
print('averted at max MDA', {k: round(float(v[-1]), 1) for k, v in G.items()})
