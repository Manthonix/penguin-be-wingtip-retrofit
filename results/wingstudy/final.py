# final outputs: named wings across all scenarios, theory comparison, whiffletree loads, csv, figures
import json, csv, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import study, rank, wing as W
from wing import Wing, RHO, G
 
os.makedirs('figures', exist_ok=True); os.makedirs('results', exist_ok=True)
C1, C2, C3, GREY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#c9c8c3', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.6, 'figure.dpi': 150,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False})
 
SEL = json.load(open('results/selection.json'))
# written by stage 2 (notebook cell 16): w2 is the best wing that passes the vortex-lattice check,
# w3 the lifting-line winner. this file runs last, so it draws the wing the screening chose
P5 = ('s', 'lam', 'tw', 'dc', 'K')
N2 = 'W2 Recommended'
N3 = 'W3 Lifting-line rank 1' + (', rejected by VLM' if SEL['W3_rejected'] else '')
REJ = f"rejected: VLM joiner {SEL['W3_row']['join38_VLM']:.3f}"
# these lifting-line figures cannot show why w3 fails, so its label carries the vortex-lattice number
NAMED = {
    'W0 Baseline':                    dict(s=1.00, lam=1.0, tw=0,   dc=0, K=0.0),
    'W1 Software-only alleviation':   dict(s=1.00, lam=1.0, tw=0,   dc=0, K=7.5),
    N2:                               dict(zip(P5, SEL['W2'])),
    N3:                               dict(zip(P5, SEL['W3'])),
    'W4 Max loiter':                  dict(s=1.175, lam=0.4, tw=-8, dc=0, K=7.5),
    'W5 Max roll authority':          dict(s=1.025, lam=0.4, tw=-8, dc=0, K=3.0),
    'W6 Twist-only extension (control)': dict(s=1.15, lam=0.4, tw=-10, dc=-6, K=0.0),
}
SC = ['passive', 'gla', 'man']
SC_LAB = {'passive': 'High-speed gust, no gust alleviation', 'gla': 'High-speed gust, with alleviation', 'man': 'Manoeuvre-sized'}
 
def mk(p):
    return Wing(s=p['s'], lam=p['lam'], tw=np.radians(p['tw']), dc=np.radians(p['dc']), K=np.radians(p['K']))
 
# ---------- named wings in every scenario ----------
bases = {}
table = []
for sc in SC:
    study.SCENARIO = sc
    b, base = study.setup()
    bases[sc] = base
    bm = base['metrics']
    for name, p in NAMED.items():
        w = mk(p); m = study.evaluate(w, base); chk = study.feasibility(m, base)
        row = dict(scenario=sc, wing=name, **p,
                   gain_cruise=100*(bm['P_cruise']/m['P_cruise']-1), gain_loiter=100*(bm['P_loiter']/m['P_loiter']-1),
                   dDi=100*(m['Di']/bm['Di']-1), dDp=100*(m['Dp']/bm['Dp']-1),
                   r_root=m['r_root'], r_join=m['r_join'], pb2v=m['pb2v'], ail_mean=m['ail_mean'],
                   m_ext=m['m_ext'], VS_mtow=m['VS_mtow'], span=w.b, S=w.S, AR=w.AR, e=m['e'],
                   tip_chord=m['tip_chord'], feasible=all(chk.values()),
                   failed='; '.join(k for k, v in chk.items() if not v),
                   worst=f"{m['worst_case_root']}/{m['worst_case_join']}")
        table.append(row)
with open('results/named_wings.csv', 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
for r in table:
    print(f"{r['scenario']:8s} {r['wing']:32s} feas={str(r['feasible']):5s} cruise {r['gain_cruise']:+6.2f}% loiter {r['gain_loiter']:+6.2f}% "
          f"Di {r['dDi']:+6.1f}% Dp {r['dDp']:+5.1f}% root {r['r_root']:.3f} join {r['r_join']:.3f} pb2v {r['pb2v']:.3f} "
          f"ail {r['ail_mean']:.3f} m {r['m_ext']:.2f} span {r['span']:.3f} {r['failed']}")
 
# ---------- all wings, csv per scenario ----------
for sc in SC:
    base, rows = rank.load(sc)
    keys = ['s', 'lam', 'tw', 'dc', 'K', 'feasible', 'gain_cruise', 'gain_loiter', 'dDi', 'margin', 'r_root', 'r_join',
            'pb2v', 'ail_mean', 'm_ext', 'VS_mtow', 'tip_chord', 'e', 'Di', 'Dp', 'P_cruise', 'P_loiter', 'V_loiter']
    with open(f'results/all_wings_{sc}.csv', 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(keys + ['failed'])
        for r in rows: wr.writerow([r[k] for k in keys] + ['; '.join(r['failed'])])
 
# ---------- theory comparison: induced drag vs span ratio ----------
e0 = bases['man']['metrics']['e']
sg = np.linspace(1.0, 1.25, 200)
ck = e0*(1 + 8*(1 - 1/sg)**2)/sg**2          # c(k=1) law: same lift, same 1 g root moment (clean-sheet)
ell = e0/sg**2                                # elliptic on the extended span, no bending limit
best = {}
for sc in SC:
    _, rows = rank.load(sc)
    ss = sorted(set(r['s'] for r in rows))
    best[sc] = [(s, min((r['dDi'] for r in rows if r['s'] == s and r['feasible']), default=np.nan)) for s in ss]
theory = dict(ck_115=100*(np.interp(1.15, sg, ck)-1), ell_115=100*(np.interp(1.15, sg, ell)-1),
              ck_1075=100*(np.interp(1.075, sg, ck)-1), ell_1075=100*(np.interp(1.075, sg, ell)-1))
print('theory', {k: round(v, 2) for k, v in theory.items()})
 
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(sg, 100*(ell-1), color=INK2, lw=1.2, ls=(0, (1, 2)), label='Elliptic on extended span (no bending limit)')
ax.plot(sg, 100*(ck-1), color=INK2, lw=1.2, ls='--', label='c(k=1) law: same 1 g root moment')
for sc, col, lw, ms in (('man', C3, 6, 0), ('gla', C1, 2.2, 5), ('passive', C2, 0, 9)):
    x, y = map(np.array, zip(*best[sc])); k = ~np.isnan(y)
    ax.plot(x[k], y[k], color=col, lw=lw, alpha=0.55 if sc == 'man' else 1, marker='o' if ms else None, ms=ms,
            solid_capstyle='round', label=f'Best feasible retrofit: {SC_LAB[sc].lower()}')
    xe, ye = x[k][-1], y[k][-1]
    ax.annotate(f'limit s = {xe:.3f}', (xe, ye), xytext=(6, -12 if sc != 'passive' else 6), textcoords='offset points', fontsize=7.5, color=INK)
ax.set_xlabel('Span ratio s = b / b₀'); ax.set_ylabel('Induced drag vs baseline (%)')
ax.set_title('How much span each structural scenario allows', loc='left', fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc='lower left'); ax.set_xlim(0.995, 1.255)
fig.savefig('figures/fig1_span_limit.png'); plt.close(fig)
 
# ---------- trade space: cruise gain vs structural margin, three panels ----------
fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
mark = {N2: ('W2', C1), N3: ('W3', C2), 'W1 Software-only alleviation': ('W1', C3)}
for ax, sc in zip(axs, SC):
    _, rows = rank.load(sc)
    inf = [r for r in rows if not r['feasible']]; F = [r for r in rows if r['feasible']]
    ax.scatter([r['gain_cruise'] for r in inf], [r['margin'] for r in inf], s=4, color=GREY, lw=0, label='Fails a constraint')
    ax.scatter([r['gain_cruise'] for r in F], [r['margin'] for r in F], s=9, color=INK2, lw=0, label='Feasible')
    pf = rank.pareto(F)
    ax.plot([r['gain_cruise'] for r in pf], [r['margin'] for r in pf], color=INK, lw=1)
    for name, (lab, col) in mark.items():
        r = next(x for x in table if x['scenario'] == sc and x['wing'] == name)
        mg = 100*(1-max(r['r_root'], r['r_join']))
        ax.scatter([r['gain_cruise']], [mg], s=46, color=col, edgecolor='white', lw=1.5, zorder=5,
                   marker='o' if r['feasible'] else 'X')
        ax.annotate(lab, (r['gain_cruise'], mg), xytext=(5, 4), textcoords='offset points', fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(SC_LAB[sc], fontsize=9, loc='left', color=INK); ax.set_xlabel('Cruise endurance gain (%)')
    ax.set_ylim(-60, 18)
axs[0].set_ylabel('Structural margin vs original rating (%)')
axs[0].legend(fontsize=7, loc='lower left', markerscale=2)
fig.suptitle('4,890 retrofit wings per scenario; ○ feasible, ✕ fails; line = Pareto front', x=0.01, ha='left', fontsize=9, color=INK2)
fig.savefig('figures/fig2_trade_space.png'); plt.close(fig)
 
# ---------- loadings: cruise and 3.8 g ----------
VA = bases['man']['VA']
def loading(w, n, V, mass, mla=True):
    q = 0.5*RHO*V**2
    d = np.clip(w.dc - (w.K if mla else 0)*(n-1), -W.DMAX, W.DMAX)
    a = w.alpha_for_CL(n*mass*G/(q*w.S), d); A = w.combine(a, d)
    return w.yf, w.lift_per_span(A, V)/(n*mass*G)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
w0, w2, w3 = mk(NAMED['W0 Baseline']), mk(NAMED[N2]), mk(NAMED[N3])
for w, lab, col, ls in ((w0, 'W0 baseline', INK2, '-'), (w2, 'W2 recommended', C1, '-'), (w3, f'W3 ({REJ})', C2, ':')):
    y, l = loading(w, 1.0, W.V_CRUISE, W.M_CRUISE + w.m_ext); m = y >= 0
    axs[0].plot(y[m], l[m], color=col, lw=2 if ls == '-' else 1.5, ls=ls, label=lab)
y, l = loading(w0, 3.8, VA, W.M_MTOW); m = y >= 0; axs[1].plot(y[m], l[m], color=INK2, lw=2, label='W0 baseline')
y, l = loading(w2, 3.8, VA, W.M_MTOW, mla=False); m = y >= 0; axs[1].plot(y[m], l[m], color=C1, lw=1.5, ls='--', label='W2, alleviation off')
y, l = loading(w2, 3.8, VA, W.M_MTOW); m = y >= 0; axs[1].plot(y[m], l[m], color=C1, lw=2, label='W2, alleviation on')
y, l = loading(w3, 3.8, VA, W.M_MTOW); m = y >= 0; axs[1].plot(y[m], l[m], color=C2, lw=1.5, ls=':', label=f'W3, alleviation on ({REJ})')
for ax, t in zip(axs, ('Cruise, 1 g, 22 m/s', '3.8 g manoeuvre at V_A')):
    ax.axvline(W.Y_CS, color=INK2, lw=0.6, ls=':'); ax.axvline(W.B0/2, color=INK2, lw=0.6, ls=':')
    ax.axvspan(*W.AIL, color=GREY, alpha=0.25, lw=0)
    ax.set_title(t, loc='left', fontsize=9, color=INK); ax.set_xlabel('Spanwise station y (m)'); ax.legend(fontsize=7.5)
axs[0].set_ylabel("Lift per span / total lift (1/m)")
axs[0].text(W.Y_CS, 0.02, ' joiner', fontsize=7, color=INK2); axs[0].text(W.B0/2, 0.02, ' original tip', fontsize=7, color=INK2)
axs[0].text(1.02, 0.36, 'ailerons', fontsize=7, color=INK2)
fig.savefig('figures/fig3_loadings.png'); plt.close(fig)
 
# ---------- root moment vs load factor (also the flight-test prediction) ----------
cap = bases['man']['cap_root']
ns = np.linspace(-1.52, 3.8, 60)
def mroot(w, n, mla=True):
    q = 0.5*RHO*VA**2
    d = np.clip(w.dc - (w.K if mla else 0)*(n-1), -W.DMAX, W.DMAX)
    a = w.alpha_for_CL(n*W.M_MTOW*G/(q*w.S), d); A = w.combine(a, d)
    return w.moment_at(w.lift_per_span(A, VA), 0.0)
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(ns, [mroot(w0, n)/cap for n in ns], color=INK2, lw=2, label='W0 baseline')
ax.plot(ns, [mroot(w2, n, mla=False)/cap for n in ns], color=C1, lw=1.5, ls='--', label='W2, alleviation off')
ax.plot(ns, [mroot(w2, n)/cap for n in ns], color=C1, lw=2, label='W2, alleviation on (recommended)')
ax.plot(ns, [mroot(w3, n)/cap for n in ns], color=C2, lw=1.5, ls=':', label=f'W3, alleviation on ({REJ})')
ax.axhline(1, color=INK, lw=0.8); ax.axhline(-1, color=INK, lw=0.8)
ax.text(-1.45, 1.03, 'original root rating (manoeuvre envelope)', fontsize=7, color=INK2)
ax.set_xlabel('Load factor n (at V_A, MTOW)'); ax.set_ylabel('Root bending / original rating')
ax.set_title('Load alleviation is what lets the longer wing reuse the joiner', loc='left', fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0, 0.93))
fig.savefig('figures/fig4_bending_vs_n.png'); plt.close(fig)
 
# ---------- weight sensitivity ----------
fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, sc in zip(axs, SC):
    o = rank.analyse(sc, verbose=False)
    ws = o['wins'][:4]
    labs = [k.replace('_', ' ').replace('s1.', 's=1.').replace('s=1.000', 's=1.00').replace(' l', ' λ').replace(' t', ' tw').replace(' d', ' δ').replace(' K', ' K') for k, _ in ws]
    ax.barh(range(len(ws))[::-1], [100*v for _, v in ws], color=C1, height=0.6)
    for i, (_, v) in enumerate(ws):
        ax.text(100*v + 1, len(ws)-1-i, f'{100*v:.0f}%', va='center', fontsize=7.5, color=INK)
    ax.set_yticks(range(len(ws))[::-1]); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlim(0, 110); ax.set_xlabel('Share of 4,000 weight draws won (%)'); ax.grid(axis='y', visible=False)
    ax.set_title(SC_LAB[sc], loc='left', fontsize=9, color=INK)
fig.tight_layout(); fig.savefig('figures/fig5_weight_sensitivity.png'); plt.close(fig)
 
# ---------- whiffletree test loads (scaled half-wing spar) ----------
# loads at the 3.8 g sizing case, same total lift, lumped into 8 equal strips over the w3 extended semispan
# (the longest wing tested, so every loading fits on one spar). the 1.000 m test spar represents w3's
# 1.8975 m semispan; total hung load 2.00 kg per half-wing. w2's load stops short of the spar tip.
SPAR, TOTAL = 1.0, 2.0
semi = w3.b/2; edges = np.linspace(0, semi, 9)
def strip_loads(w, mla):
    y, l = loading(w, 3.8, VA, W.M_MTOW, mla=mla); m = y >= 0; y, l = y[m], l[m]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (y >= a) & (y <= b)
        if k.sum() < 2: out.append((0.5*(a+b), 0.0)); continue
        F = np.trapezoid(l[k], y[k]); yc = np.trapezoid(l[k]*y[k], y[k])/F if abs(F) > 1e-12 else 0.5*(a+b)
        out.append((yc, F))
    tot = sum(F for _, F in out)
    return [(yc/semi*SPAR, F/tot*TOTAL) for yc, F in out]
wt = {'W0 baseline': strip_loads(w0, True), 'W2 alleviation off': strip_loads(w2, False), 'W2 alleviation on': strip_loads(w2, True),
      'W3 alleviation off': strip_loads(w3, False), 'W3 alleviation on': strip_loads(w3, True)}
with open('results/whiffletree_loads.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['loading', 'strip', 'position_on_1m_spar_m', 'hung_mass_kg'])
    for name, L in wt.items():
        for i, (x, mkg) in enumerate(L): wr.writerow([name, i+1, f'{x:.3f}', f'{mkg:.3f}'])
jx = W.Y_CS/semi*SPAR
for name, L in wt.items():
    Mr = sum(x*mkg*G for x, mkg in L); Mj = sum((x-jx)*mkg*G for x, mkg in L if x > jx)
    print(f"{name:20s} root {Mr:.3f} N m  joiner({jx:.3f} m) {Mj:.3f} N m  | " + '  '.join(f'{x:.3f}:{m:.3f}' for x, m in L))
json.dump(dict(theory=theory, VA=VA, cap_root=cap, cap_join=bases['man']['cap_join'], e0=e0,
               whiffletree={k: v for k, v in wt.items()}), open('results/summary.json', 'w'), indent=1, default=float)
