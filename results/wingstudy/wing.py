# penguin be retrofit wing model: prandtl lifting line (all harmonics, full span),
# naca 2412 section data from neuralfoil, load cases, bending, roll, stall.
# units: si, angles in radians internally.
import numpy as np
from airfoil import Polars
 
RHO, NU, G = 1.225, 1.46e-5, 9.81
POL = Polars()
 
# ---------- baseline aircraft (penguin be electric datasheet + inferred planform) ----------
B0 = 3.3            # baseline span, m
Y_CS = 0.55         # centre-section half-length = joiner station, m
C_ROOT0, C_TIP0 = 0.288, 0.163   # inferred from the datasheet dimension drawing (gives 0.813 m2)
_AREA_INF = 2*(Y_CS*C_ROOT0 + (B0/2 - Y_CS)*(C_ROOT0 + C_TIP0)/2)
C_ROOT, C_TIP = (C_ROOT0*0.79/_AREA_INF, C_TIP0*0.79/_AREA_INF)   # scaled to the published 0.79 m2
M_CRUISE = 17.7     # kg, empty 14.9 + 2.8 payload (endurance mission)
M_MTOW = 21.5       # kg
V_CRUISE = 22.0     # m/s
AIL = (1.00, 1.60)  # aileron span on each side, m (assumed; not published)
TAU = 0.50          # aileron effectiveness for 25% chord (assumed)
DMAX = np.radians(20.0)   # aileron travel limit
RHO_A = 3.0         # extension areal mass, kg/m^2 (assumed, composite skin + spar)
M_FIT = 0.04        # per-side attachment fitting, kg (assumed)
 
 
class Wing:
    """one retrofit design: s = span ratio, lam = extension taper, tw = extension tip twist (rad),
    dc = symmetric aileron trim in cruise (rad, +ve trailing-edge down), K = mla gain (rad per g)."""
 
    def __init__(self, s=1.0, lam=1.0, tw=0.0, dc=0.0, K=0.0, N=80, nfine=2001):
        self.s, self.lam, self.tw, self.dc, self.K = s, lam, tw, dc, K
        self.b = B0 * s
        self.dext = (self.b - B0) / 2           # extension length per side
        self.N = N
        # collocation (open interval) and fine grid for integrals
        self.th = np.arange(1, N + 1) * np.pi / (N + 1)
        self.thf = np.linspace(1e-4, np.pi - 1e-4, nfine)
        self.yc = -self.b / 2 * np.cos(self.th)
        self.yf = -self.b / 2 * np.cos(self.thf)
        self.cc, self.cf = self.chord(self.yc), self.chord(self.yf)
        # area by fine integration
        self.S = np.trapezoid(self.cf, self.yf)
        self.AR = self.b**2 / self.S
        self.m_ext = 0.0 if self.dext <= 0 else 2 * (RHO_A * self.ext_area_half() + M_FIT)
        self._setup()
 
    # ---------- geometry ----------
    def chord(self, y):
        a = np.abs(y)
        c = np.where(a <= Y_CS, C_ROOT, C_ROOT + (C_TIP - C_ROOT) * (a - Y_CS) / (B0 / 2 - Y_CS))
        if self.dext > 0:
            t = np.clip((a - B0 / 2) / self.dext, 0, 1)
            c = np.where(a > B0 / 2, C_TIP * (1 + (self.lam - 1) * t), c)
        return c
 
    def ext_area_half(self):
        return self.dext * C_TIP * (1 + self.lam) / 2
 
    def twist(self, y):
        # zero twist on the original wing (it cannot be re-twisted); linear 0 -> tw on the extension
        a = np.abs(y)
        if self.dext <= 0:
            return np.zeros_like(a)
        return np.where(a > B0 / 2, self.tw * np.clip((a - B0 / 2) / self.dext, 0, 1), 0.0)
 
    def ail(self, y):
        a = np.abs(y)
        return ((a >= AIL[0]) & (a <= AIL[1])).astype(float)
 
    # ---------- lifting line ----------
    def _setup(self):
        # section data at the local cruise re (fixed per design)
        re_c = V_CRUISE * self.cc / NU
        self.a0c, self.al0c, _, _ = POL.lin(re_c)
        n = np.arange(1, self.N + 1)
        S, T = np.meshgrid(n, self.th)
        mu = self.a0c * self.cc / (4 * self.b)
        self.Amat = np.sin(S * T) * (np.sin(T) + S * mu[:, None])
        self.mu = mu
        # unit right-hand sides -> fourier coefficients (linear superposition)
        rhs = {
            'alpha': np.ones_like(self.th),                       # uniform +1 rad
            'zero':  self.twist(self.yc) - self.al0c,             # twist minus zero-lift angle
            'dsym':  TAU * self.ail(self.yc),                     # +1 rad symmetric aileron
            'dasym': TAU * self.ail(self.yc) * np.sign(self.yc),  # +1 rad right te-down, left te-up
            'proll': 2 * self.yc / self.b,                        # unit pb/2V, right wing down -> right wing alpha up
        }
        self.coef = {k: np.linalg.solve(self.Amat, mu * v * np.sin(self.th)) for k, v in rhs.items()}
        self.n = n
        self.sinf = np.sin(np.outer(self.thf, n))
 
    def combine(self, alpha, dsym=0.0, dasym=0.0, proll=0.0):
        c = self.coef
        return alpha * c['alpha'] + c['zero'] + dsym * c['dsym'] + dasym * c['dasym'] + proll * c['proll']
 
    def CL(self, A):
        return np.pi * self.AR * A[0]
 
    def CDi(self, A):
        return np.pi * self.AR * np.sum(self.n * A**2)
 
    def alpha_for_CL(self, CLreq, dsym):
        A0 = self.combine(0.0, dsym)
        return (CLreq - self.CL(A0)) / (np.pi * self.AR * self.coef['alpha'][0])
 
    def gamma_hat(self, A):
        # gamma / (2 b V) on fine grid
        return self.sinf @ A
 
    def lift_per_span(self, A, V):
        return RHO * V**2 * 2 * self.b * self.gamma_hat(A)
 
    def cl_local(self, A, y=None):
        return 4 * self.b * self.gamma_hat(A) / self.cf
 
    def moment_at(self, Lp, y0):
        # bending moment about station y0 on the right wing (positive = lift up)
        m = self.yf >= y0
        return np.trapezoid(Lp[m] * (self.yf[m] - y0), self.yf[m])
 
    def roll_coef(self, A):
        # rolling moment coefficient, positive right wing down (from lift distribution)
        Lhat = self.gamma_hat(A)   # proportional to lift per span
        return -np.trapezoid(Lhat * self.yf, self.yf) * 2 * self.b / (self.S * self.b) * 2
 
    # ---------- performance ----------
    def section_cd(self, A, V):
        re = V * self.cf / NU
        return POL.cd(self.cl_local(A), re)
 
    def drag(self, mass, V, dsym=None):
        """wing drag at level flight; returns dict of components (n)."""
        dsym = self.dc if dsym is None else dsym
        q = 0.5 * RHO * V**2
        CLreq = mass * G / (q * self.S)
        a = self.alpha_for_CL(CLreq, dsym)
        A = self.combine(a, dsym)
        Di = q * self.S * self.CDi(A)
        Dp = q * np.trapezoid(self.cf * self.section_cd(A, V), self.yf)
        return dict(CL=CLreq, alpha=a, A=A, Di=Di, Dp=Dp, Dw=Di + Dp, e=CLreq**2 / (np.pi * self.AR * self.CDi(A)))
 
    def stall(self, V=17.0, dsym=0.0):
        """critical-section stall: wing CL when the first station reaches its 2-d clmax.
        returns CLmax, first-stall station |y|, max cl/clmax over the aileron span at that moment."""
        re = V * self.cf / NU
        _, _, clmax, _ = POL.lin(re)
        A0, A1 = self.combine(0.0, dsym), self.coef['alpha']
        c0, c1 = self.cl_local(A0), self.cl_local(A1)
        # cl(alpha) = c0 + alpha*c1 ; find smallest alpha where any station hits clmax
        with np.errstate(divide='ignore', invalid='ignore'):
            ah = np.where(c1 > 0, (clmax - c0) / c1, np.inf)
        j = np.argmin(ah)
        a = ah[j]
        A = A0 + a * A1
        ratio = self.cl_local(A) / clmax
        ail = self.ail(self.yf) > 0
        return dict(CLmax=self.CL(A), y_first=abs(self.yf[j]), ail_ratio=ratio[ail].max(),
                    ail_mean=ratio[ail].mean())
 
    def roll_authority(self):
        # steady roll with remaining differential travel after cruise trim
        d_avail = max(DMAX - abs(self.dc), 0.0)
        Cld = self.roll_coef(self.coef['dasym'])
        Clp = self.roll_coef(self.coef['proll'])
        return abs(Cld * d_avail / Clp), Cld, Clp
 
    def CL_limit(self, V, dsym, sign=+1):
        """wing CL at which the first section reaches its 2-d clmax (sign +1) or clmin (sign -1)."""
        re = V * self.cf / NU
        _, _, clmax, clmin = POL.lin(re)
        lim = clmax if sign > 0 else clmin
        A0, A1 = self.combine(0.0, dsym), self.coef['alpha']
        c0, c1 = self.cl_local(A0), self.cl_local(A1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ah = np.where(c1 > 0, (lim - c0) / c1, np.inf if sign > 0 else -np.inf)
        a = np.min(ah) if sign > 0 else np.max(ah)
        return self.CL(A0 + a * A1)
 
    # ---------- structural load cases ----------
    def loads(self, cases, gla=False):
        """bending moments at root and joiner for each case. case = dict(kind, n, V, mass, U).
        gla=True: the autopilot answers gusts with the same quasi-steady up-aileron schedule as manoeuvres
        (optimistic: a real gust system has lag)."""
        out = []
        for c in cases:
            q = 0.5 * RHO * c['V']**2
            if c['kind'] == 'man':
                # manoeuvre: mla schedules symmetric up-aileron with load factor (+ve K = more up per g)
                d = np.clip(self.dc - self.K * (c['n'] - 1.0), -DMAX, DMAX)
                CLreq = c['n'] * c['mass'] * G / (q * self.S)
                a = self.alpha_for_CL(CLreq, d)
                A = self.combine(a, d)
            else:
                # gust: 1 g trim at V plus a sharp-edged gust increment, no alleviation (conservative)
                CL1 = c['mass'] * G / (q * self.S)
                a1 = self.alpha_for_CL(CL1, self.dc)
                A1 = self.combine(a1, self.dc)
                CLa = np.pi * self.AR * self.coef['alpha'][0]
                cbar = self.S / self.b
                mug = 2 * (c['mass'] * G / self.S) / (RHO * cbar * CLa * G)
                Kg = 0.88 * mug / (5.3 + mug)
                dalpha = Kg * c['U'] / c['V']
                # a gust cannot push the wing past stall: cap total CL at the critical-section CLmax / CLmin
                CLlim = self.CL_limit(c['V'], self.dc, +1 if c['U'] > 0 else -1)
                CLg = CL1 + CLa * dalpha
                if (c['U'] > 0 and CLg > CLlim) or (c['U'] < 0 and CLg < CLlim):
                    dalpha = (CLlim - CL1) / CLa
                A = A1 + dalpha * self.coef['alpha']
                if gla and self.K > 0:
                    n_g = np.pi * self.AR * A[0] * q * self.S / (c['mass'] * G)
                    d = np.clip(self.dc - self.K * (n_g - 1.0), -DMAX, DMAX)
                    A = A + (d - self.dc) * self.coef['dsym']
            Lp = self.lift_per_span(A, c['V'])
            out.append(dict(name=c['name'], Mroot=self.moment_at(Lp, 0.0), Mjoin=self.moment_at(Lp, Y_CS),
                            Mmid=self.moment_at(Lp, 1.10), Mtipjoint=self.moment_at(Lp, B0 / 2),
                            n_eff=np.pi * self.AR * A[0] * q * self.S / (c['mass'] * G)))
        return out
