# naca 2412 polars from neuralfoil, cached to disk
import numpy as np, os, neuralfoil as nf
 
def naca4(m=0.02, p=0.4, t=0.12, n=121):
    # cosine-spaced naca 4-digit, upper te -> le -> lower te (selig order)
    beta = np.linspace(0, np.pi, n)
    x = 0.5*(1 - np.cos(beta))
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return np.vstack([np.c_[xu[::-1], yu[::-1]], np.c_[xl[1:], yl[1:]]])
 
RE = np.geomspace(4e4, 1.2e6, 16)
ALPHA = np.arange(-12, 20.01, 0.25)
CACHE = os.path.join(os.path.dirname(__file__), 'naca2412_polars.npz')
 
def build():
    xy = naca4()
    CL = np.zeros((len(RE), len(ALPHA))); CD = np.zeros_like(CL); CONF = np.zeros_like(CL)
    for i, re in enumerate(RE):
        r = nf.get_aero_from_coordinates(coordinates=xy, alpha=ALPHA, Re=re, model_size='xlarge')
        CL[i], CD[i], CONF[i] = r['CL'], r['CD'], r['analysis_confidence']
    np.savez(CACHE, RE=RE, ALPHA=ALPHA, CL=CL, CD=CD, CONF=CONF)
 
class Polars:
    # per-re: linear-range slope a0 and alpha_l0, clmax/clmin, and cd(cl) on the attached branch
    def __init__(self):
        if not os.path.exists(CACHE): build()
        d = np.load(CACHE)
        self.RE, self.A, self.CL, self.CD = d['RE'], d['ALPHA'], d['CL'], d['CD']
        self.logRE = np.log(self.RE)
        lin = (self.A >= -4) & (self.A <= 8)   # wide band averages out low-re bubble wiggles
        self.a0, self.al0, self.clmax, self.clmin = [], [], [], []
        self.branch = []
        for i in range(len(self.RE)):
            k, b = np.polyfit(np.radians(self.A[lin]), self.CL[i, lin], 1)
            k = min(k, 2*np.pi)   # >2pi is a bubble artefact; cap at thin-airfoil value
            self.a0.append(k); self.al0.append(-b/k)
            imax = np.argmax(self.CL[i]); imin = np.argmin(self.CL[i, :imax+1])
            self.clmax.append(self.CL[i, imax]); self.clmin.append(self.CL[i, imin])
            cl_b, cd_b = self.CL[i, imin:imax+1], self.CD[i, imin:imax+1]
            # enforce monotone cl on the branch for interpolation
            keep = np.r_[True, np.diff(np.maximum.accumulate(cl_b)) > 1e-6]
            self.branch.append((np.maximum.accumulate(cl_b)[keep], cd_b[keep]))
        self.a0, self.al0 = np.array(self.a0), np.array(self.al0)
        self.clmax, self.clmin = np.array(self.clmax), np.array(self.clmin)
        # dense (re, cl) -> cd table, so profile drag is a single vectorised interpolation
        from scipy.interpolate import RegularGridInterpolator
        self.clg = np.arange(-2.0, 2.5001, 0.005)
        tab = np.empty((len(self.RE), len(self.clg)))
        for i, (clb, cdb) in enumerate(self.branch):
            v = np.interp(self.clg, clb, cdb)
            hi, lo = self.clg > clb[-1], self.clg < clb[0]
            v[hi] = cdb[-1] + 0.5*(self.clg[hi]-clb[-1])**2 + 0.05*(self.clg[hi]-clb[-1])
            v[lo] = cdb[0] + 0.5*(clb[0]-self.clg[lo])**2 + 0.05*(clb[0]-self.clg[lo])
            tab[i] = v
        self._cdi = RegularGridInterpolator((self.logRE, self.clg), tab, bounds_error=False, fill_value=None)
 
    def _w(self, re):
        x = np.clip(np.log(re), self.logRE[0], self.logRE[-1])
        j = np.clip(np.searchsorted(self.logRE, x) - 1, 0, len(self.RE) - 2)
        t = (x - self.logRE[j]) / (self.logRE[j+1] - self.logRE[j])
        return j, t
 
    def lin(self, re):
        j, t = self._w(re)
        f = lambda a: a[j]*(1-t) + a[j+1]*t
        return f(self.a0), f(self.al0), f(self.clmax), f(self.clmin)
 
    def cd(self, cl, re):
        # cd at local cl and re; beyond the attached branch, stall cd plus a steep penalty
        x = np.clip(np.log(re), self.logRE[0], self.logRE[-1])
        c = np.clip(cl, self.clg[0], self.clg[-1])
        return self._cdi(np.c_[x, c])
 
if __name__ == '__main__':
    build(); P = Polars()
    for re in (1e5, 2.5e5, 3.6e5, 4.3e5):
        a0, al0, cmx, cmn = P.lin(np.array([re]))
        cd = P.cd(np.array([0.0, 0.4, 0.8]), np.array([re]*3))
        print(f"Re={re:.1e}  a0={a0[0]:.3f}/rad ({a0[0]*np.pi/180:.4f}/deg)  aL0={np.degrees(al0[0]):.2f} deg  clmax={cmx[0]:.3f}  cd(0,0.4,0.8)={np.round(cd,5)}")
