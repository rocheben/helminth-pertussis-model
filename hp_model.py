"""
Helminth / Bordetella pertussis model, structured by age and worm burden
(burden structure after Fenton 2013, Parasitology 140:1119-1132).

Hosts are classified by
  age class      0-3 months, 3-12 months, 1-14 years, 15 years and over
                 (mean lifespan 50 years, stationary population)
  pertussis      S0 never infected nor vaccinated, SW susceptible again after
                 waning, I infectious, C convalescent, R immune after infection,
                 V protected by vaccination, WN waning phase (exposure boosts
                 back to R at rate nu * lambda, otherwise -> SW)
  worm burden    j = 0..J (hosts aged 1 year and over only)
  exposure group g (gamma-distributed exposure to helminth infective stages,
                 shape kappa, which generates aggregated worm burdens)

Helminths: as in Fenton (2013), infection pressure is a property of the setting
not altered by pertussis; hosts acquire worms at rate lam_w * g * e_s (e_s = rho
for I and C), each worm dies at rate 1 + m (m = clearance added by mass drug
administration). lam_w is calibrated to the helminth prevalence among children.

Interaction, for a host carrying j worms (both saturating, Fenton 2013):
  susceptibility   phi(j) = 1 + (sigma - 1) j / (w50 + j)
  infectiousness   tau(j) = 1 + (tau - 1) j / (w50 + j)
w50 = w50f x mean burden of infected hosts aged 1 year and over, untreated.

Vaccination at 3 months (single event standing for the primary series), with
efficacy eps and coverage cov. Total protection Tv years after vaccination and
Tn after infection; the waning phase lasts min(Tn/2, 0.75 Tv).
Pertussis mixing between three contact groups (infants, children, adults) is
homogeneous, except that child-child contacts are multiplied by acc; beta is
set so that R0 is reached in a helminth-free, fully susceptible population.

For given forces of infection the model is linear, so equilibria are solved
exactly as sparse linear systems and the forces of infection by root finding.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import roots_genlaguerre, gamma as G_
from scipy.optimize import brentq, root

DUR = np.array([0.25, 0.75, 14.0, 35.0])
MU = 1 / DUR.sum()
GAMMA_P, GAMMA_C, MU_W = 365 / 21.0, 12.0, 1.0
S0, SW, I, C, R, V, WN = range(7)
NS = 7
CGROUP = np.array([0, 0, 1, 2])          # contact group of each age class

BASE = dict(tau=1.0, sigma=10.0, w50f=1.0, prev=0.40, kappa=0.5, R0=12.0, cov=0.72,
            eps=0.90, Tv=30.0, Tn=30.0, nu=1.0, acc=1.0, rho=0.5, m=0.0)


class Model:
    def __init__(self, P, J=None, NG=8):
        self.P = dict(BASE, **P)
        p = self.P
        self.NG = NG
        if J is None:
            J = 80
        self.J = J
        x_, w_ = roots_genlaguerre(NG, p['kappa'] - 1)
        self.GRP = x_ / p['kappa']; self.WGT = w_ / G_(p['kappa'])
        self.jj = np.arange(J + 1)
        self.n_inf = 2 * NG * NS
        self.N = self.n_inf + 2 * NG * NS * (J + 1)
        # durations
        Tw = min(p['Tn'] / 2, 0.75 * p['Tv'])
        self.om_W = 1 / Tw
        self.om_V = 1 / max(p['Tv'] - Tw, 1e-3)
        self.om_R = 1 / max(p['Tn'] - Tw, 1e-3)

    # ------------------------------------------------------------ indexing
    def idx(self, a, g, s, j=0):
        if a < 2:
            return (a * self.NG + g) * NS + s
        return self.n_inf + (((a - 2) * self.NG + g) * NS + s) * (self.J + 1) + j

    # ------------------------------------------------------ helminths only
    def helminth_children(self, lam_w, m):
        """burden distribution of children (1-14) and of all hosts >= 1 yr, no pertussis"""
        NG, J = self.NG, self.J
        n = 2 * NG * (J + 1)
        rows, cols, vals = [], [], []
        b = np.zeros(n)
        def ix(a, g, j): return ((a * NG + g) * (J + 1)) + j
        for a in range(2):
            out_age = 1 / DUR[a + 2]
            for g in range(NG):
                ai = lam_w * self.GRP[g]
                k = ix(a, g, self.jj)
                out = np.full(J + 1, out_age)
                if a == 0:
                    rows += list(ix(1, g, self.jj)); cols += list(k); vals += [out_age] * (J + 1)
                kk = k[:-1]
                rows += list(kk + 1); cols += list(kk); vals += [ai] * J
                out[:-1] += ai
                d = self.jj[1:] * (MU_W + m)
                rows += list(k[1:] - 1); cols += list(k[1:]); vals += list(d)
                out[1:] += d
                rows += list(k); cols += list(k); vals += list(-out)
                b[ix(0, g, 0)] = MU * self.WGT[g]
        A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        x = spla.spsolve(A, -b).reshape(2, NG, J + 1)
        bc = x[0].sum(axis=0) / x[0].sum()
        ba = x.sum(axis=(0, 1)) / x.sum()
        return bc, ba, x

    def calibrate(self):
        p = self.P
        f = lambda l: 1 - self.helminth_children(l, 0.0)[0][0] - p['prev']
        self.lam_w = brentq(f, 1e-4, 1e3, xtol=1e-8)
        bc, ba, _ = self.helminth_children(self.lam_w, 0.0)
        mean_all = (self.jj * ba).sum(); prev_all = 1 - ba[0]
        self.w50 = p['w50f'] * mean_all / prev_all
        mean_c = (self.jj * bc).sum(); var_c = (self.jj ** 2 * bc).sum() - mean_c ** 2
        self.k_hat = mean_c ** 2 / (var_c - mean_c) if var_c > mean_c else np.inf
        self.tail = max(bc[-1], ba[-1])
        return self

    # -------------------------------------------------------------- build
    def build(self, sigma, m):
        p, NG, J, jj = self.P, self.NG, self.J, self.jj
        phi = 1 + (sigma - 1) * jj / (self.w50 + jj)
        rows, cols, vals = [], [], []
        A1 = [([], [], []) for _ in range(3)]
        expo_s = np.ones(NS); expo_s[I] = p['rho']; expo_s[C] = p['rho']
        trans = {I: (C, GAMMA_P), C: (R, GAMMA_C), R: (WN, self.om_R),
                 V: (WN, self.om_V), WN: (SW, self.om_W)}
        def add(r, c, v):
            r = np.atleast_1d(r); c = np.atleast_1d(c)
            rows.extend(r); cols.extend(np.broadcast_to(c, r.shape)); vals.extend(np.broadcast_to(np.asarray(v, float), r.shape))
        for a in range(4):
            age_out = 1 / DUR[a]
            cg = CGROUP[a]
            for g in range(NG):
                for s in range(NS):
                    if a < 2:
                        k = self.idx(a, g, s); out = age_out
                        if a == 0 and s == S0:
                            add([self.idx(1, g, V), self.idx(1, g, S0)], [k, k],
                                [age_out * p['eps'] * p['cov'], age_out * (1 - p['eps'] * p['cov'])])
                        elif a == 0:
                            add(self.idx(1, g, s), k, age_out)
                        else:
                            add(self.idx(2, g, s, 0), k, age_out)
                        if s in trans:
                            t, r_ = trans[s]; out += r_; add(self.idx(a, g, t), k, r_)
                        if s == WN:
                            A1[cg][0].extend([k, self.idx(a, g, R)]); A1[cg][1].extend([k, k]); A1[cg][2].extend([-p['nu'], p['nu']])
                        if s in (S0, SW):
                            A1[cg][0].extend([k, self.idx(a, g, I)]); A1[cg][1].extend([k, k]); A1[cg][2].extend([-1.0, 1.0])
                        add(k, k, -out)
                        continue
                    k = self.idx(a, g, s, jj)
                    out = np.full(J + 1, age_out if a == 2 else 1 / DUR[3])
                    if a == 2:
                        add(self.idx(3, g, s, jj), k, age_out)
                    ai = self.lam_w * self.GRP[g] * expo_s[s]
                    add(k[:-1] + 1, k[:-1], ai); out[:-1] += ai
                    d = jj[1:] * (MU_W + m); add(k[1:] - 1, k[1:], d); out[1:] += d
                    if s in trans:
                        t, r_ = trans[s]; out += r_; add(self.idx(a, g, t, jj), k, r_)
                    if s == WN:
                        A1[cg][0].extend(list(k) + list(self.idx(a, g, R, jj))); A1[cg][1].extend(list(k) * 2)
                        A1[cg][2].extend([-p['nu']] * (J + 1) + [p['nu']] * (J + 1))
                    if s in (S0, SW):
                        A1[cg][0].extend(list(k) + list(self.idx(a, g, I, jj))); A1[cg][1].extend(list(k) * 2)
                        A1[cg][2].extend(list(-phi) + list(phi))
                    add(k, k, -out)
        N = self.N
        A0 = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        A1m = [sp.csc_matrix((v, (r, c)), shape=(N, N)) for r, c, v in A1]
        b = np.zeros(N)
        for g in range(NG):
            b[self.idx(0, g, S0)] = MU * self.WGT[g]
        return A0, A1m, b, phi

    # -------------------------------------------------------- equilibrium
    def masks(self):
        if hasattr(self, '_masks'): return self._masks
        N, NG, J = self.N, self.NG, self.J
        age = np.zeros(N, int); stat = np.zeros(N, int); burden = np.zeros(N, int)
        for a in range(4):
            for g in range(NG):
                for s in range(NS):
                    if a < 2:
                        k = self.idx(a, g, s); age[k] = a; stat[k] = s
                    else:
                        k = self.idx(a, g, s, self.jj); age[k] = a; stat[k] = s; burden[k] = self.jj
        self._masks = (age, stat, burden); return self._masks

    def equilibrium(self, sigma, m=None):
        p = self.P; m = p['m'] if m is None else m
        A0, A1, b, phi = self.build(sigma, m)
        age, stat, burden = self.masks()
        Cm = np.ones((3, 3)); Cm[1, 1] = p['acc']
        Ng = np.array([DUR[:2].sum(), DUR[2], DUR[3]]) * MU
        K0 = Cm * Ng[:, None]
        beta = p['R0'] * GAMMA_P / max(abs(np.linalg.eigvals(K0)))
        grp = CGROUP[age]
        tau_n = np.where(age >= 2, 1 + (p['tau'] - 1) * burden / (self.w50 + burden), 1.0)
        Ig = lambda x: np.array([(x * tau_n)[(stat == I) & (grp == q)].sum() for q in range(3)])
        def solve(lam):
            M = A0.copy()
            for q in range(3): M = M + lam[q] * A1[q]
            return spla.spsolve(M.tocsc(), -b)
        # disease-free: Re from next-generation matrix with susceptibility weights
        x0 = solve(np.zeros(3))
        phi_n = np.where(age >= 2, phi[burden], 1.0)
        susw = np.array([(x0 * phi_n)[((stat == S0) | (stat == SW)) & (grp == q)].sum() for q in range(3)])
        susw_tau = np.array([(x0 * phi_n * tau_n)[((stat == S0) | (stat == SW)) & (grp == q)].sum() for q in range(3)])
        tau_g = np.where(susw > 0, susw_tau / np.maximum(susw, 1e-300), 1.0)
        Re = max(abs(np.linalg.eigvals(beta / GAMMA_P * Cm * susw[:, None] * tau_g[None, :])))
        sus_all = ((stat == S0) | (stat == SW))
        delta = (x0 * phi_n)[sus_all].sum() / x0[sus_all].sum()
        if Re <= 1:
            return dict(Re=Re, delta=delta, lam=np.zeros(3), x=x0, phi_n=phi_n, endemic=False)
        F = lambda z: beta * Cm @ Ig(solve(np.exp(z))) - np.exp(z)
        Fl = lambda z: np.log(np.maximum(beta * Cm @ Ig(solve(np.exp(z))), 1e-300)) - z
        z0 = np.log(np.full(3, 0.5))
        sol = root(Fl, z0, method='hybr', options=dict(xtol=1e-10))
        lam = np.exp(sol.x)
        x = solve(lam)
        return dict(Re=Re, delta=delta, lam=lam, x=x, phi_n=phi_n, endemic=True, ok=sol.success,
                    resid=np.abs(F(sol.x)).max())

    def outcomes(self, eq):
        age, stat, burden = self.masks()
        x, phi_n, lam = eq['x'], eq['phi_n'], eq['lam']
        grp = CGROUP[age]
        foi = lam[grp] * phi_n
        sus = (stat == S0) | (stat == SW)
        inc = foi * x * sus
        size = np.array([x[age == a].sum() for a in range(4)])
        inf_inf = inc[age < 2].sum() / size[:2].sum() * 1000
        inf_03 = inc[age == 0].sum() / size[0] * 1000
        child_first = inc[(age == 2) & (stat == S0)].sum() / size[2] * 1000
        child_all = inc[age == 2].sum() / size[2] * 1000
        tot = inc.sum() * 1000
        worm_share = inc[burden > 0].sum() / inc.sum() if inc.sum() > 0 else np.nan
        return dict(infants=inf_inf, inf03=inf_03, child_first=child_first,
                    child_all=child_all, total=tot, worm_share=worm_share)
