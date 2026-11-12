"""Global sensitivity analysis (Latin hypercube) of the helminth/pertussis model.
Outcome: relative change in pertussis infections caused by the interaction
(sigma vs sigma = 1, same parameters otherwise).
Usage: python sensitivity.py N_SAMPLES OUTPUT_CSV   (N_SAMPLES = 500 in the Appendix)"""
import sys, json, numpy as np, csv
from multiprocessing import Pool
from scipy.stats import qmc
from hp_model import Model

RANGES = dict(tau=(1, 5, 'log'), sigma=(2, 100, 'log'), w50f=(0.2, 5, 'log'), prev=(0.2, 0.7, 'lin'),
              kappa=(0.1, 0.5, 'lin'), R0=(5, 17, 'lin'), cov=(0.5, 0.95, 'lin'),
              eps=(0.84, 0.94, 'lin'), Tv=(20, 50, 'lin'), Tn=(30, 70, 'lin'),
              nu=(0, 3, 'lin'), acc=(1, 4, 'lin'))

def sample(n, seed=1):
    u = qmc.LatinHypercube(d=len(RANGES), seed=seed).random(n)
    out = []
    for row in u:
        p = {}
        for (k, (lo, hi, sc)), v in zip(RANGES.items(), row):
            p[k] = float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo)))) if sc == 'log' else float(lo + v * (hi - lo))
        out.append(p)
    return out

def make_model(p):
    J = 80
    while True:
        M = Model(p, J=J).calibrate()
        if M.tail < 1e-6 or J >= 640: return M
        J *= 2

def run(p):
    try:
        M = make_model(p)
        tau = M.P['tau']; M.P['tau'] = 1.0
        e1 = M.equilibrium(1.0); o1 = M.outcomes(e1)
        M.P['tau'] = tau
        e2 = M.equilibrium(p['sigma']); o2 = M.outcomes(e2)
        r = dict(p, J=M.J, tail=M.tail, k_hat=M.k_hat, Re1=e1['Re'], Re2=e2['Re'], delta=e2['delta'],
                 ok=bool(e1.get('ok', True) and e2.get('ok', True)))
        for k in o1:
            r[k + '_1'] = o1[k]; r[k + '_s'] = o2[k]
        return r
    except Exception as ex:
        return dict(p, error=str(ex))

if __name__ == '__main__':
    n = int(sys.argv[1]); out = sys.argv[2]
    ps = sample(n)
    with Pool(2) as pool:
        res = pool.map(run, ps, chunksize=4)
    keys = sorted({k for r in res for k in r})
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in res]
    print('done', len(res), sum('error' in r for r in res))
