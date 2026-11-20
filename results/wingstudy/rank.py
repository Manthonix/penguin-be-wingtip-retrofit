# ranking: single-objective champions, priority-weighted balanced wing, weight sensitivity, pareto front
import json, numpy as np, sys
 
# objective: (key, +1 maximise / -1 minimise, label)
OBJ = [('gain_cruise', +1, 'Cruise endurance gain (%)'),
       ('gain_loiter', +1, 'Loiter endurance gain (%)'),
       ('margin',      +1, 'Structural margin vs original rating (%)'),
       ('pb2v',        +1, 'Roll authority pb/2V'),
       ('ail_mean',    -1, 'Aileron c_l/c_l,max at stall onset'),
       ('m_ext',       -1, 'Added mass (kg)'),
       ('VS_mtow',     -1, 'Clean stall speed at MTOW (m/s)')]
 
# priority weights for an endurance-retrofit mission (sum = 1)
W_PRIORITY = dict(gain_cruise=0.30, gain_loiter=0.20, margin=0.20, pb2v=0.10, ail_mean=0.10, m_ext=0.05, VS_mtow=0.05)
W_ALT = {
    'Endurance-first': dict(gain_cruise=0.45, gain_loiter=0.30, margin=0.10, pb2v=0.05, ail_mean=0.05, m_ext=0.03, VS_mtow=0.02),
    'Structure-first': dict(gain_cruise=0.15, gain_loiter=0.10, margin=0.50, pb2v=0.10, ail_mean=0.05, m_ext=0.05, VS_mtow=0.05),
    'Handling-first':  dict(gain_cruise=0.15, gain_loiter=0.10, margin=0.15, pb2v=0.25, ail_mean=0.25, m_ext=0.05, VS_mtow=0.05),
    'Equal weights':   {k: 1/7 for k, _, _ in OBJ},
}
 
def key(r):
    return f"s{r['s']:.3f}_l{r['lam']:.1f}_t{r['tw']:+.0f}_d{r['dc']:+.0f}_K{r['K']:.1f}"
 
def load(sc):
    d = json.load(open(f'rows_{sc}.json'))
    return d['base'], d['rows']
 
def norm_matrix(F):
    M = np.array([[r[k] * sgn for k, sgn, _ in OBJ] for r in F], float)
    lo, hi = M.min(0), M.max(0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (M - lo) / span
 
def score(F, w):
    Z = norm_matrix(F)
    wv = np.array([w[k] for k, _, _ in OBJ])
    return Z @ wv
 
def pareto(F, a='gain_cruise', b='margin'):
    pts = sorted(F, key=lambda r: (-r[a], -r[b]))
    front, best = [], -np.inf
    for r in pts:
        if r[b] > best + 1e-9:
            front.append(r); best = r[b]
    return front
 
def analyse(sc, n_mc=4000, seed=1, verbose=True):
    base, rows = load(sc)
    F = [r for r in rows if r['feasible']]
    out = dict(scenario=sc, n_total=len(rows), n_feasible=len(F))
    # champions
    champs = {}
    for k, sgn, lab in OBJ + [('dDi', -1, 'Induced drag change (%)')]:
        best = max(F, key=lambda r: (sgn * r[k], r['gain_cruise']))
        champs[lab] = best
    out['champions'] = champs
    # balanced
    sc_p = score(F, W_PRIORITY)
    order = np.argsort(-sc_p)
    out['ranked'] = [(F[i], sc_p[i]) for i in order[:10]]
    out['balanced'] = F[order[0]]
    # named alternative weightings
    out['alt'] = {name: F[int(np.argmax(score(F, w)))] for name, w in W_ALT.items()}
    # monte-carlo weights around the priority set
    rng = np.random.default_rng(seed)
    alpha = 30 * np.array([W_PRIORITY[k] for k, _, _ in OBJ])
    Z = norm_matrix(F)
    wins = {}
    for _ in range(n_mc):
        wv = rng.dirichlet(alpha)
        i = int(np.argmax(Z @ wv))
        wins[key(F[i])] = wins.get(key(F[i]), 0) + 1
    out['wins'] = sorted(((k, v / n_mc) for k, v in wins.items()), key=lambda x: -x[1])
    out['pareto'] = pareto(F)
    if verbose:
        report(out)
    return out
 
def fmt(r):
    return (f"s={r['s']:.3f} taper={r['lam']:.1f} tw={r['tw']:+.0f}° trim={r['dc']:+.0f}° K={r['K']:.1f}°/g | "
            f"cruise {r['gain_cruise']:+.2f}%  loiter {r['gain_loiter']:+.2f}%  Di {r['dDi']:+.1f}%  "
            f"margin {r['margin']:+.1f}%  pb/2V {r['pb2v']:.3f}  ail {r['ail_mean']:.3f}  "
            f"m+{r['m_ext']:.2f}kg  Vs {r['VS_mtow']:.2f}")
 
def report(o):
    print(f"\n######## scenario: {o['scenario']}  ({o['n_feasible']} of {o['n_total']} feasible)")
    print("-- single-objective champions")
    for lab, r in o['champions'].items():
        print(f"  {lab:42s} {fmt(r)}")
    print("-- top 5 under priority weights")
    for r, s in o['ranked'][:5]:
        print(f"  {s:.3f}  {fmt(r)}")
    print("-- winners under alternative weightings")
    for n, r in o['alt'].items():
        print(f"  {n:16s} {fmt(r)}")
    print("-- monte-carlo win share (4000 weight draws around the priority set)")
    for k, v in o['wins'][:5]:
        print(f"  {v*100:5.1f}%  {k}")
    print(f"-- pareto front (cruise gain vs margin): {len(o['pareto'])} wings")
 
if __name__ == '__main__':
    for sc in (sys.argv[1:] or ['passive', 'gla', 'man']):
        analyse(sc)
