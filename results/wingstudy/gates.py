# validation gates for the lifting-line model. run after any change to wing.py.
import numpy as np, wing
from wing import Wing
 
def ell_wing(AR=8.0, N=80, a0=2*np.pi):
    # temporary elliptic, untwisted, constant-a0 wing built by overriding geometry
    w = Wing(s=1.0, N=N)
    b = w.b; c0 = 4*b/(np.pi*AR)
    w.chord = lambda y: c0*np.sqrt(np.clip(1-(2*y/b)**2, 0, None))
    w.twist = lambda y: np.zeros_like(y)
    w.cc, w.cf = w.chord(w.yc), w.chord(w.yf)
    w.S = np.trapezoid(w.cf, w.yf); w.AR = b**2/w.S
    w.al0_override = True
    # constant section data
    re_save = wing.POL.lin
    wing.POL.lin = lambda re: (np.full_like(re, a0), np.zeros_like(re), np.full_like(re, 1.3), np.full_like(re, -1.0))
    w._setup()
    wing.POL.lin = re_save
    return w
 
ok = True
def gate(name, val, ref, tol):
    global ok
    p = abs(val-ref) <= tol*abs(ref) if ref != 0 else abs(val) <= tol
    ok &= p
    print(f"{'PASS' if p else 'FAIL'}  {name}: {val:.6g} (ref {ref:.6g}, tol {tol:g})")
 
# g1: elliptic planform -> elliptic loading, e = 1, lift slope 2pi/(1+2/AR)
w = ell_wing(AR=8)
A = w.combine(np.radians(5))
gate("G1a elliptic: max|A_n>1|/A1", np.max(np.abs(A[1:]))/A[0], 0.0, 2e-3)
gate("G1b elliptic: CL_alpha", w.CL(w.coef['alpha']), 2*np.pi/(1+2/8.0), 2e-3)
CL = w.CL(A)
gate("G1c elliptic: e", CL**2/(np.pi*w.AR*w.CDi(A)), 1.0, 1e-3)
# g2: root bending of elliptic loading = L b/(3 pi); joiner-station ratio vs closed form
V = 20.0; Lp = w.lift_per_span(A, V); L = np.trapezoid(Lp, w.yf)
gate("G2a elliptic root moment / (L b / 3pi)", w.moment_at(Lp, 0)/(L*w.b/(3*np.pi)), 1.0, 2e-3)
eta0 = 2*0.55/w.b
ref = 3*((1-eta0**2)**1.5/3 - eta0*(np.pi/4 - (eta0*np.sqrt(1-eta0**2)+np.arcsin(eta0))/2))
gate("G2b elliptic M(y0)/M_root", w.moment_at(Lp, 0.55)/w.moment_at(Lp, 0), ref, 3e-3)
# g3: roll damping of elliptic wing, a0 = 2pi, lifting line closed form Clp = -(pi AR/8)*... check vs strip theory at huge AR
w2 = ell_wing(AR=400)
Clp = w2.roll_coef(w2.coef['proll'])
gate("G3 elliptic Clp at AR=400 vs strip theory -a0/8", Clp, -2*np.pi/8, 1e-2)
# g4: convergence of the real baseline in N
d60 = Wing(N=60).drag(17.7, 22.0); d100 = Wing(N=100).drag(17.7, 22.0)
gate("G4a baseline Di convergence N=60 vs 100", d60['Di'], d100['Di'], 5e-3)
L60 = Wing(N=60).loads([dict(name='m', kind='man', n=3.8, V=30, mass=21.5)])[0]['Mroot']
L100 = Wing(N=100).loads([dict(name='m', kind='man', n=3.8, V=30, mass=21.5)])[0]['Mroot']
gate("G4b baseline root moment convergence", L60, L100, 3e-3)
# g5: lift by integration equals the required lift
b = Wing(); d = b.drag(17.7, 22.0); Lp = b.lift_per_span(d['A'], 22.0)
gate("G5 integrated lift / weight", np.trapezoid(Lp, b.yf)/(17.7*9.81), 1.0, 2e-3)
# g6: c(k) cross-check - for an untwisted extended wing the root moment / lift lever grows ~ with span
w15 = Wing(s=1.15, lam=1.0); d15 = w15.drag(17.7, 22.0)
lev = lambda w, d: w.moment_at(w.lift_per_span(d['A'], 22.0), 0)/(17.7*9.81)
print(f"info  untwisted s=1.15 root-lever ratio = {lev(w15,d15)/lev(b,d):.4f} (first-order guess was 1.15)")
print(f"info  baseline: S={b.S:.4f} m2  AR={b.AR:.3f}  CL={d['CL']:.3f}  e={d['e']:.4f}  Di={d['Di']:.3f} N  Dp={d['Dp']:.3f} N  wing L/D={17.7*9.81/d['Dw']:.2f}")
print("ALL GATES PASS" if ok else "SOME GATES FAILED")
