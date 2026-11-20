# retrofit trade study: sweep the design space, apply the mission constraints,
# find single-objective champions and the priority-weighted balanced wing.
import numpy as np, itertools, json, time
import wing as W
from wing import Wing, RHO, G
 
LD_AIRCRAFT = 18.0      # assumed baseline whole-aircraft cruise l/d (sensitivity run at 15 and 20)
GUST_U = 15.0           # assumed cs-vla-style gust, m/s
V_GUST = 32.4           # gust case at 0.9 v_h (conservative); lift capped at stall, so a 22 m/s gust only reaches 1.38 g
SCENARIO = 'gla'        # 'passive' | 'gla' | 'man'
 
def cases(VA):
    c = [dict(name='M+3.8', kind='man', n=3.8, V=VA, mass=W.M_MTOW),
         dict(name='M-1.52', kind='man', n=-1.52, V=VA, mass=W.M_MTOW)]
    if SCENARIO != 'man':
        c += [dict(name='G+15', kind='gust', U=+GUST_U, V=V_GUST, mass=W.M_MTOW),
              dict(name='G-15', kind='gust', U=-GUST_U, V=V_GUST, mass=W.M_MTOW)]
    return c
 
def setup(ld=LD_AIRCRAFT):
    b = Wing()
    st = b.stall()
    VS1 = np.sqrt(2 * W.M_MTOW * G / (RHO * b.S * st['CLmax']))
    VA = VS1 * np.sqrt(3.8)
    d = b.drag(W.M_CRUISE, W.V_CRUISE)
    q = 0.5 * RHO * W.V_CRUISE**2
    D_tot = W.M_CRUISE * G / ld
    CDnw_S = (D_tot - d['Dw']) / q            # non-wing drag area (fuselage, tail, booms), m^2
    L = b.loads(cases(VA), gla=(SCENARIO == 'gla'))
    cap_root = max(abs(x['Mroot']) for x in L)
    cap_join = max(abs(x['Mjoin']) for x in L)
    base = dict(VA=VA, VS1=VS1, CDnw_S=CDnw_S, cap_root=cap_root, cap_join=cap_join,
                stall_y=st['y_first'], CLmax=st['CLmax'], loads=L)
    base['metrics'] = evaluate(b, base, quick=False)
    return b, base
 
def power(w, mass, V, CDnw_S):
    d = w.drag(mass, V)
    D = d['Dw'] + 0.5 * RHO * V**2 * CDnw_S
    return D * V, D, d
 
def evaluate(w, base, quick=True):
    m_cr = W.M_CRUISE + w.m_ext           # retrofit flies the same mission with the extension on board
    P, D, d = power(w, m_cr, W.V_CRUISE, base['CDnw_S'])
    st = w.stall()
    VS = np.sqrt(2 * W.M_MTOW * G / (RHO * w.S * st['CLmax']))
    # loiter: minimum power over speed, flown no slower than 1.25 vs at the cruise mass
    vs_cr = np.sqrt(2 * m_cr * G / (RHO * w.S * st['CLmax']))
    Vg = np.arange(max(1.25 * vs_cr, 12.0), 30.01, 0.5)
    Pg = np.array([power(w, m_cr, v, base['CDnw_S'])[0] for v in Vg])
    i = np.argmin(Pg)
    L = w.loads(cases(base['VA']), gla=(SCENARIO == 'gla'))
    r_root = max(abs(x['Mroot']) for x in L) / base['cap_root']
    r_join = max(abs(x['Mjoin']) for x in L) / base['cap_join']
    pb2v = w.roll_authority()[0]
    return dict(P_cruise=P, D_total=D, Di=d['Di'], Dp=d['Dp'], e=d['e'], CL=d['CL'],
                P_loiter=Pg[i], V_loiter=Vg[i], VS_mtow=VS, CLmax=st['CLmax'], stall_y=st['y_first'], ail_mean=st['ail_mean'],
                r_root=r_root, r_join=r_join, tip_chord=float(w.chord(np.array([w.b/2]))[0]), pb2v=pb2v, m_ext=w.m_ext, dext=w.dext, S=w.S, AR=w.AR,
                worst_case_root=max(L, key=lambda x: abs(x['Mroot']))['name'],
                worst_case_join=max(L, key=lambda x: abs(x['Mjoin']))['name'])
 
def feasibility(m, base):
    bm = base['metrics']
    chk = {
        'A1 root': m['r_root'] <= 1.0 + 1e-9,
        'A2 joiner': m['r_join'] <= 1.0 + 1e-9,
        'B3 stall speed': m['VS_mtow'] <= bm['VS_mtow'] + 1e-6,
        'B4 aileron not fully stalled': m['ail_mean'] <= 0.99,
        'B5 roll': m['pb2v'] >= 0.07,
        'C2 ext length': m['dext'] <= 0.35 + 1e-9,
        'C3 added mass': m['m_ext'] <= 0.40 + 1e-9,
        'C7 tip chord >= 60 mm': m['tip_chord'] >= 0.060 - 1e-9,
    }
    return chk
 
GRID = dict(s=[1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.175, 1.20, 1.25],
            lam=[1.0, 0.7, 0.4],
            tw=[0, -2, -4, -6, -8, -10],          # deg, extension tip twist
            dc=[2, 0, -2, -4, -6],                # deg, cruise symmetric aileron (- = up)
            K=[0, 1.5, 3.0, 4.5, 6.0, 7.5])       # deg per g, mla gain (travel clips at 20 deg)
 
def run(ld=LD_AIRCRAFT, verbose=True, scenario=None):
    global SCENARIO
    if scenario: SCENARIO = scenario
    b, base = setup(ld)
    rows = []
    t = time.time()
    for s, lam, tw, dc, K in itertools.product(*GRID.values()):
        if s == 1.0 and (lam != 1.0 or tw != 0):
            continue                          # no extension -> extension variables are meaningless
        w = Wing(s=s, lam=lam, tw=np.radians(tw), dc=np.radians(dc), K=np.radians(K))
        m = evaluate(w, base)
        chk = feasibility(m, base)
        m.update(s=s, lam=lam, tw=tw, dc=dc, K=K, feasible=all(chk.values()),
                 failed=[k for k, v in chk.items() if not v])
        rows.append(m)
    bm = base['metrics']
    for m in rows:
        m['gain_cruise'] = 100 * (bm['P_cruise'] / m['P_cruise'] - 1)    # endurance change at cruise, %
        m['gain_loiter'] = 100 * (bm['P_loiter'] / m['P_loiter'] - 1)    # endurance change at best-endurance speed, %
        m['dDi'] = 100 * (m['Di'] / bm['Di'] - 1)
        m['margin'] = 100 * (1 - max(m['r_root'], m['r_join']))           # structural margin vs original rating, %
    if verbose:
        print(f"{len(rows)} wings evaluated in {time.time()-t:.1f} s; feasible: {sum(r['feasible'] for r in rows)}")
    return b, base, rows
 
if __name__ == '__main__':
    import sys
    b, base, rows = run(scenario=sys.argv[1] if len(sys.argv) > 1 else None)
    bm = base['metrics']
    print(f"baseline: VA={base['VA']:.2f} m/s  VS1={base['VS1']:.2f}  CLmax={base['CLmax']:.3f}  CDnw*S={base['CDnw_S']*1e4:.1f} cm2")
    print("baseline loads (N m):", [(x['name'], round(x['Mroot'], 1), round(x['Mjoin'], 1)) for x in base['loads']])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in bm.items()})
    with open(f'rows_{SCENARIO}.json', 'w') as f:
        json.dump(dict(base={k: v for k, v in base.items() if k != 'loads'} | {'loads': base['loads']}, rows=rows), f, default=float)
