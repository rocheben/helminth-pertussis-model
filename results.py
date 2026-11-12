"""Numbers quoted in the manuscript and Appendix A3. Requires hp_model.py.
Usage: python results.py OUTPUT_JSON"""
import sys, json, numpy as np
from scipy.optimize import brentq
from hp_model import Model, BASE, CGROUP

CENTRAL = dict(BASE, sigma=10.0, tau=2.0)

def model(**kw):
    return Model(dict(CENTRAL, **kw)).calibrate()

def m_for_prev(M, target):
    return brentq(lambda m: 1 - M.helminth_children(M.lam_w, m)[0][0] - target, 0, 200, xtol=1e-6)

def worm_prev(M, eq):
    age, stat, burden = M.masks(); x = eq['x']
    return 1 - x[(age >= 2) & (burden == 0)].sum() / x[age >= 2].sum()

def run(M, sigma, tau, m=0.0):
    keep = M.P['tau']; M.P['tau'] = tau
    e = M.equilibrium(sigma, m=m); o = M.outcomes(e); M.P['tau'] = keep
    o.update(Re=e['Re'], delta=e['delta'], worm_prev=worm_prev(M, e)); return o

out = {}
M = model()
out['k_hat'] = M.k_hat; out['w50'] = M.w50
m20 = m_for_prev(M, 0.20); out['m20'] = m20
out['prev_m20b'] = 1 - M.helminth_children(M.lam_w, 20.0)[0][0]
ref = run(M, 1.0, 1.0); out['ref'] = ref
R0 = CENTRAL['R0']
for sig, tau, lab in ((1, 1, 'none'), (10, 1, 'sus'), (10, 2, 'central'), (100, 1, 'sus100'), (2, 1, 'sus2')):
    for mlab, m in (('u', 0.0), ('m20', m20), ('m20b', 20.0)):
        o = run(M, sig, tau, m)
        o['mult'] = o['Re'] / ref['Re']
        o['req'] = 1 - 1 / (R0 * o['mult'])
        out[f'{lab}_{mlab}'] = o
for tau in (1.0, 1.5, 2.0, 3.0, 5.0):
    u = run(M, 10, tau); d = run(M, 10, tau, m20)
    out[f'tau{tau}'] = dict(rel_inf=(u['infants'] / ref['infants'] - 1) * 100, inf=u['infants'],
                            dew20=d['infants'], dew_gain=(1 - d['infants'] / u['infants']) * 100,
                            req=1 - 1 / (R0 * u['Re'] / ref['Re']))
# vaccination effect in the central scenario
M0 = model(cov=0.0)
out['central_cov0_inf'] = run(M0, 10, 2)['infants']; out['none_cov0_inf'] = run(M0, 1, 1)['infants']
# delta from prevalence alone (sigma = 10): every susceptible carrying worms gets sigma
age, stat, burden = M.masks()
import scipy.sparse.linalg as spla
A0, A1, b, _ = M.build(1.0, 0.0); x0 = spla.spsolve(A0.tocsc(), -b); sus = (stat <= 1)
out['delta_gen10'] = 1 + 9 * (x0 * (burden / (M.w50 + burden)))[sus].sum() / x0[sus].sum()
out['delta_prev10'] = 1 + 9 * x0[sus & (burden > 0)].sum() / x0[sus].sum()
# reverse effect: helminth prevalence (1 yr+) with rho vs without (rho = 1), no vaccination
for rho in (0.1, 0.5, 0.9):
    a = run(model(cov=0.0, rho=rho), 10, 2)['worm_prev']; b = run(model(cov=0.0, rho=1.0), 10, 2)['worm_prev']
    out[f'rev_rho{rho}'] = (a - b) * 100
# re at full coverage, no interaction
out['Re_cov1_none'] = run(model(cov=1.0), 1, 1)['Re']
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'results.json', 'w'), indent=1, default=float)
def r(v): return round(float(v), 3)
print(json.dumps(out, indent=1, default=r))
