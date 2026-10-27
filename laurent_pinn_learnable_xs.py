"""
Laurent-Enriched PINN with a LEARNABLE Singularity Position
===========================================================

Extension of the fixed-pole Laurent decomposition scheme to the case where the
singularity location x_s is UNKNOWN a priori and must be discovered during
training. Designed as the method core for a journal paper.

Solution ansatz (general pole order alpha, learnable position):
    u(x) = U_smooth(x; theta) + P(x; A, alpha, x_s)

    hard pole : P = -A * (x_s - x)^(-alpha)              (blowing side)
                P = -A * cos(pi*alpha) * |x_s - x|^(-alpha)  (far side;
                for alpha=1 this is exactly A/(x - x_s))
    soft pole : P = -A * Re[ (x_s - x + i*eps)^(-alpha) ]
              = -A * r^(-alpha) * cos(alpha*theta),
              r = sqrt(delta^2 + eps^2), theta = atan2(eps, delta),
              delta = x_s - x
    (A, alpha) are FIXED per equation class by dominant balance (see the
    universality note below): Riccati u^2 -> A=-1, alpha=1; cubic u^3 ->
    A=-1/sqrt(2), alpha=1/2. Only x_s is learnable.

Method components
-----------------
1. Hybrid analytic gradient (anti-explosion core, inherited from the fixed-pole
   scheme): du/dx = autograd(U_smooth, x) + closed_form(dP/dx). Autograd never
   differentiates the pole term w.r.t. the input coordinate, which cuts the
   exploding backward path that breaks vanilla PINNs near singularities.

2. Learnable pole position and residue:
       x_s = w_lo + (w_hi - w_lo) * sigmoid(z)   (search-window constrained)
   The window [w_lo, w_hi] may extend beyond the region of physical interest,
   which also covers edge singularities sitting on the domain boundary.

3. Complex-plane pole annealing:
   eps(t) follows a cosine schedule from eps0 down to eps_min. A large eps
   keeps |P|, |dP/dx| and all parameter gradients bounded early in training;
   eps -> eps_min recovers exact pole asymptotics. Equivalently, the pole is
   lowered from the complex plane onto the real axis.

4. Two-phase collocation sampling:
   Points are resampled every epoch. Phase R (refinement): log-dense on BOTH
   sides of the current pole estimate (distance log-uniform in [ball, span])
   plus a uniform background, rejecting points within
   ball = max(c*eps, ball_min) of x_s; in hard-pole mode this ball stays
   active in both phases for numerical safety. Phase D (discovery): pure
   uniform sampling over the whole window with NO exclusion ball -- the
   pinned soft pole keeps P, dP and every gradient bounded, and unbiased
   coverage avoids self-confirming density around a wrong estimate. This
   also fixes the sign error of the original near-singularity sampler
   (which landed points in [-1.12, 0.57] instead of near pi/2).

5. Roughness-guided discovery (identifiability restoration):
   For Riccati-type ODEs the pure pole term A/(x - x_s) satisfies the residual
   IDENTICALLY for every x_s (R = -A(1+A)/(x-x_s)^2 = 0 when A = -1): the
   solution family coincides with the ansatz, so the free smooth branch can
   absorb the initial condition and produce a manifold of compensating local
   minima (observed: x_s stalls at a wrong location). We restore
   identifiability with an annealed roughness penalty w(t) * mean(s'^2) on
   the smooth branch: compensating s must be spiky (s' ~ 1/eps^2), while the
   true smooth remainder is O(1). Training is three-phase:
     Phase D (discovery)  : uniform collocation (no self-confirming bias),
                            eps pinned at eps0, roughness weight w0, residue
                            A frozen at the class-determined value A* = -1
                            (unfrozen uniform sampling otherwise kills the
                            pole via its O(A^2/eps^2) local residual before
                            x_s can migrate).
     Phase R1 (refinement): same PRE-POLE window, cosine-anneal
                            eps -> eps_min, A still frozen, K still active --
                            all collocation points sit on the true flow, so
                            K stays consistent and polishes the position.
     Phase R2 (final fit) : FULL window, A unfrozen (residue recovered from
                            pure physics: residual ~ -(A+1)/d^2 near the
                            pole), log-dense sampling + exclusion ball, K OFF.
   The roughness weight is constant: the true Laurent remainder is smooth
   (identically zero for pure blow-up), so the penalty only bites spiky
   absorbers hiding a misplaced pole.

6. Phase-integral discovery functional (direct position gradient, GENERAL):
   Any penalty acting on the smooth branch alone has EXACTLY ZERO gradient
   w.r.t. x_s (s is a separate network that never sees x_s), so the position
   is identifiable only through weak indirect coupling through the physics
   residual -- observed as x_s moving ~0.01 over 3000 epochs. The general
   discovery principle comes from the singularity's own defining property:
   an autonomous flow u' = f(u) blows up in finite time iff the PHASE
   INTEGRAL (time-to-blow-up) converges (Osgood criterion),
       tau(u) = \int_u^{\infty} ds / f(s)   <  infinity,
   and every trajectory point satisfies the exact identity
       x_s = x + tau(u(x))        (on the whole maximal interval),
   i.e. in the phase plane (x, G(u)), G(u) = \int^u ds/f(s), the solution
   is a straight line of slope 1 whose x-intercept is the pole. Discovery
   is therefore a regression problem with loss
       K(x_s) = mean_col[ rho( x + tau(u(x)) - x_s ) ]   (rho = Tukey biweight,
   redescending: rho'(w) = 0 for |w| > 1, so disconnected-branch votes --
   e.g. the analytic continuation of tan past its pole votes 3*pi/2 --
   are rejected exactly instead of being clamped like Huber's),
   where tau is computed (analytically or by quadrature) from the KNOWN
   equation f -- no residue, no pole order, no solution form assumed.
   Properties:
   (a) consistency: on any exact flow the residual is identically 0 at the
       true pole and K(x_s) = (x_s - x*)^2 -- a pure quadratic in x_s with
       no spurious local minima, on both sides of the pole;
   (b) unit-weight votes: the residual is O(1) even as |u| -> infinity,
       killing the u^2-amplified "sag-vote bias" of the earlier Riccati
       moment form mean[(u(x - x_s) + 1)^2] (which over-weighted the
       post-pole side and caused the observed 0.99 -> 1.28 overshoot);
   (c) built-in falsification: non-blow-up solutions satisfy the identity
       with the pole of their ANALYTIC CONTINUATION outside the physical
       window (control u = -1/(1+x): x + 1/u = -1 exactly), so a no-pole
       verdict is a sharp, verifiable "virtual pole" reading, not a
       degenerate A -> 0 collapse;
   (d) the earlier pole-moment functional is exactly the f = u^2 special
       case (tau = 1/u): u(x - x_s) + 1 = 0 <=> x + 1/u = x_s.
   K is active only where every collocation point sits on the true flow
   (Phases D and R1, pre-pole windows) and OFF in the final full-window
   fit (an exact displaced flow also satisfies the identity at ITS pole,
   so post-pole votes can self-confirm a misestimate).

7. Hard initial-condition constraint (lazy-mode elimination):
   For u' = u^2 the ZERO function solves the ODE identically, so soft-IC
   collocation training admits a LAZY solution: u ~ 0 everywhere plus a thin
   bump near x_0 that fakes u(x_0) = u_0 (cost ~ O(1) physics loss). The
   blow-up is then never attempted and the discovery functional sees no
   spike to grab. We therefore impose the IC EXACTLY by output
   reparameterization (standard hard-constraint trick):
       u(x) = P(x; A, x_s) + [u_0 - P(x_0; A, x_s)] + (x - x_0) * NN(x)
   which satisfies u(x_0) = u_0 identically in (A, x_s); any deviation from
   the true IVP flow now costs O(u^2) residual EVERYWHERE, making the true
   solution the unique cheap minimizer.

8. Window continuation (flow-basin forcing):
   Even with a hard IC the network can CRASH from u_0 to ~0 inside a narrow
   band right after x_0 -- on a long window the band is sparsely sampled and
   the crash is cheap, so training lands in the lazy basin and the true flow
   (hence the blow-up the pole-moment functional needs) is never attempted.
   Phase D therefore trains on a GROWING window [x_0, T_k], T_k = T_0 * g^k:
   on short windows collocation density is high, the crash costs O((u')^2)
   over many points and the flow-following solution is the only cheap
   minimum; each extension warm-starts from the previous fit. The
   phase-integral functional locks x_s along the way because the identity
   x + tau(u(x)) = x_s holds at EVERY point of the exact flow, not only
   near the pole -- K is quadratic in x_s on every sub-window.
   Two safeguards keep the K votes unbiased during growth:
   (a) ERROR GATE -- the window advances only after the fit on the current
       window has converged (physics-loss EMA below a tolerance), never
       onto a still-sagging fit;
   (b) EPS-AWARE ADAPTIVE CAP -- T never crosses the current pole estimate
       (T <= x_s_hat - max(margin, 1.5*eps0 + 0.02)), so collocation stays
       out of the soft pole's steep zone |d| < eps and past-pole regions;
       a time-driven schedule with a bare 0.05 margin was observed to let
       x_s snowball from 0.99 to 1.28 through the sag-vote bias.

Benchmarks
----------
tan      u' = u^2 + 1, u(0)=0  -> tan(x),          pole at pi/2    (interior)
blowup   u' = u^2,     u(0)=1  -> 1/(1-x),         pole at x=1     (interior)
control  u' = u^2,     u(0)=-1 -> -1/(1+x),        virtual pole at x=-1
         (no blow-up inside [0,2]; the phase integral still votes for the
          pole of the analytic continuation OUTSIDE the window -- a sharp
          falsification target once the search window includes it)
cubic    u' = u^3,     u(0)=1  -> (1-2x)^(-1/2),   pole at x=1/2, ALPHA = 1/2
         (edge blow-up with a different pole ORDER: tests that the ansatz
          and the functional generalize beyond the Riccati class)

Universality of the singularity parameters: for an autonomous ODE
u' = f(u) with f(u) ~ c*u^p as u -> infinity, dominant balance
    B* alpha (x_s - x)^(-alpha-1) = c B*^p (x_s - x)^(-p alpha)
forces the pole ORDER alpha* = 1/(p-1) and the residue B*^(p-1) = alpha*/c.
The SHAPE of the singularity is therefore determined by the equation:
for u^p (c=1): p=2 -> alpha=1, B*=1 (Riccati, A*=-B*=-1);
               p=3 -> alpha=1/2, B*=1/sqrt(2) (cubic benchmark).
Only the POSITION x_s is unknown and must be discovered -- via the
phase-integral identity, which likewise needs nothing but f itself.
All three singularity degrees of freedom are equation-determined.

Usage examples
--------------
python laurent_pinn_learnable_xs.py --bench blowup --xs-init 0.7
python laurent_pinn_learnable_xs.py --suite multistart --bench blowup
python laurent_pinn_learnable_xs.py --bench control   # virtual pole: should land at x_s = -1
python laurent_pinn_learnable_xs.py --bench cubic     # different pole order alpha = 1/2
python laurent_pinn_learnable_xs.py --bench tan --vanilla --clip 0
python laurent_pinn_learnable_xs.py --bench blowup --pole-k0 0   # ablation: no phase functional
"""

import argparse
import math
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================= Reproducibility & device =================
def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

seed_everything(1314)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad).to(device)


# ================= Utility: Xavier init for linear layers =================
def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


# ================= Input normalization (smooth branch only) =================
class CoordNormalizer:
    """MinMax map [x_min, x_max] -> [-1, 1] with exact inverse."""

    def __init__(self, x_min: float, x_max: float):
        self.x_min, self.x_max = x_min, x_max

    def forward(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0


# ================= Baseline: vanilla PINN (control group) =================
class VanillaPINN(nn.Module):
    """Smooth Tanh MLP with LayerNorm, MinMax-normalized input.
    Hard-IC output transform: u(x) = u_0 + (x - x_0) * net(x)."""

    def __init__(self, hidden_layers=4, hidden_units=32, x_min=0.0, x_max=2.0,
                 x_bc=0.0, u_bc=0.0, hard_ic=True):
        super().__init__()
        self.normalizer = CoordNormalizer(x_min, x_max)
        self.x0, self.u0 = float(x_bc), float(u_bc)
        self.hard_ic = bool(hard_ic)
        layers = [nn.Linear(1, hidden_units), nn.LayerNorm(hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.LayerNorm(hidden_units), nn.Tanh()]
        layers += [nn.Linear(hidden_units, 1)]
        self.net = nn.Sequential(*layers)
        self.net.apply(xavier_init)

    def forward(self, x):
        out = self.net(self.normalizer.forward(x))
        if self.hard_ic:
            return self.u0 + (x - self.x0) * out
        return out


# ================= Proposed model: Laurent PINN, learnable pole =================
class LaurentPINN(nn.Module):
    """
    u(x) = P(x; A, alpha, x_s) + s(x)
    s(x) = [u_0 - P(x_0)] + (x - x_0) * NN(x)      (hard IC: u(x_0) = u_0 exact)

    A     : learnable residue (scalar); frozen at the dominant-balance value
            A* during discovery, recovered from pure physics in the final fit
    alpha : pole ORDER, fixed per equation class by dominant balance
            (alpha* = 1/(p-1) for f ~ c u^p) -- never learnable
    x_s   : learnable pole position via sigmoid parameterization over the
            SEARCH window [s_lo, s_hi], which is DECOUPLED from (and may
            extend beyond) the physical collocation window so virtual poles
            outside the domain remain reachable
    eps   : annealing height of the pole above the real axis (plain float,
            controlled by the trainer). eps == 0 -> exact hard pole.
    """

    def __init__(self, hidden_layers=4, hidden_units=32, window=(0.0, 2.0),
                 search=None, xs_init=0.8, A_init=-1.0, alpha=1.0,
                 learn_xs=True, learn_A=True, soft_pole=True, x_bc=0.0, u_bc=0.0):
        super().__init__()
        self.w_lo, self.w_hi = float(window[0]), float(window[1])
        s = window if search is None else search
        self.s_lo, self.s_hi = float(s[0]), float(s[1])
        self.normalizer = CoordNormalizer(self.w_lo, self.w_hi)
        self.x0, self.u0 = float(x_bc), float(u_bc)
        self.alpha = float(alpha)

        layers = [nn.Linear(1, hidden_units), nn.LayerNorm(hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.LayerNorm(hidden_units), nn.Tanh()]
        layers += [nn.Linear(hidden_units, 1)]
        self.smooth_net = nn.Sequential(*layers)
        self.smooth_net.apply(xavier_init)

        # Sigmoid parameterization keeps x_s inside the SEARCH window
        # [s_lo, s_hi] and yields smooth, bounded position gradients.
        frac = (float(xs_init) - self.s_lo) / (self.s_hi - self.s_lo)
        frac = min(max(frac, 0.02), 0.98)
        z0 = math.log(frac / (1.0 - frac))
        # z is NOT a differentiable parameter: the pole position is updated by
        # the damped-Newton location step on the phase-vote Huber objective
        # (train loop), never by backprop. Physics/roughness/BC therefore
        # CANNOT steer the pole -- a misplaced interior pole would otherwise
        # give them a free wrong-branch flow to soften by dragging x_s away
        # from the target (observed runaway 0.8 -> +1.54 on control).
        self.z = nn.Parameter(torch.tensor(z0, dtype=torch.float32, device=device),
                              requires_grad=False)
        self.A = nn.Parameter(torch.tensor(float(A_init), dtype=torch.float32, device=device),
                              requires_grad=learn_A)
        self.learn_xs = bool(learn_xs)
        self.learn_A = bool(learn_A)
        self.soft_pole = soft_pole
        self.eps = 0.12 if soft_pole else 0.0

    @property
    def xs(self):
        return self.s_lo + (self.s_hi - self.s_lo) * torch.sigmoid(self.z)

    def set_xs(self, xs_new: float):
        """Clamp into the search window and reparameterize to z (no-grad)."""
        span = self.s_hi - self.s_lo
        frac = min(max((float(xs_new) - self.s_lo) / span, 1e-3), 1.0 - 1e-3)
        with torch.no_grad():
            self.z.fill_(math.log(frac / (1.0 - frac)))

    def set_eps(self, eps: float):
        self.eps = float(eps)

    def _pole(self, d):
        """d = x_s - x (signed distance to the pole). Return (P, dP/dx) for
        the current eps; closed-form, autograd never differentiates this
        branch w.r.t. the input coordinate.

        soft pole (eps > 0), complex-power regularization:
            P = -A * Re[(d + i eps)^(-alpha)] = -A * r^(-alpha) cos(alpha*th),
            r = sqrt(d^2 + eps^2), th = atan2(eps, d) in (0, pi)
            dP/dx = -A*alpha*r^(-alpha-2) * (d*cos(alpha*th) - eps*sin(alpha*th))
        hard pole (eps = 0), exact power + far-side analytic continuation:
            P = -A * d^(-alpha)                     (d > 0, blowing side)
            P = -A * cos(pi*alpha) * |d|^(-alpha)   (d < 0, far side; for
                alpha = 1 this is exactly A/(x - x_s))
        """
        if self.soft_pole and self.eps > 0:
            e = self.eps
            r = torch.sqrt(d * d + e * e)
            th = torch.atan2(torch.full_like(d, e), d)
            c = torch.cos(self.alpha * th)
            s = torch.sin(self.alpha * th)
            p = -self.A * r ** (-self.alpha) * c
            dp = -self.A * self.alpha * r ** (-self.alpha - 2.0) * (d * c - e * s)
        else:
            ad = d.abs() + 1e-12
            cpa = math.cos(math.pi * self.alpha)
            pow_p = ad ** (-self.alpha)
            pow_dp = ad ** (-self.alpha - 1.0)
            p = torch.where(d > 0, -self.A * pow_p, -self.A * cpa * pow_p)
            dp = torch.where(d > 0, -self.A * self.alpha * pow_dp,
                             self.A * self.alpha * cpa * pow_dp)
        return p, dp

    def _smooth_branch(self, x):
        """Hard-IC smooth branch: s(x) = [u_0 - P(x_0)] + (x - x_0)*NN(x).
        The constant re-centers the pole so u(x_0) = u_0 holds EXACTLY for
        every (A, x_s, eps) -- the IC bump (lazy mode) is structurally gone."""
        nn_out = self.smooth_net(self.normalizer.forward(x))
        p0, _ = self._pole(self.xs - self.x0)
        return (self.u0 - p0) + (x - self.x0) * nn_out

    def forward(self, x):
        s = self._smooth_branch(x)
        p, _ = self._pole(self.xs - x)
        return s + p

    def get_u_and_du(self, x):
        """Hybrid derivative: autograd for the smooth branch w.r.t. raw x,
        closed-form for the pole branch. This is the anti-explosion core.
        Returns (u, du, ds): ds is the smooth-branch derivative, reused by the
        roughness penalty at zero extra autograd cost."""
        s = self._smooth_branch(x)
        ds = torch.autograd.grad(s.sum(), x, create_graph=True, retain_graph=True)[0]
        p, dp = self._pole(self.xs - x)
        return s + p, ds + dp, ds


# ================= Benchmark suite =================
# Each benchmark provides:
#   rhs       : f(u) for u' = f(u)
#   exact     : closed-form solution (numpy)
#   H         : PHASE INTEGRAL tau(u) = \int_u^{+inf} ds / f(s) (torch),
#               the time-to-blow-up; finite iff the flow blows up. Guards
#               only protect transient fits, never the true flow.
#   window    : physical collocation domain [w_lo, w_hi]
#   search    : sigmoid search window for x_s (may extend outside the
#               physical window, e.g. to reach a virtual pole)
#   xs_true   : pole of the maximal analytic continuation (for control it
#               sits OUTSIDE the window -- the falsification target)
#   A_true    : residue in the convention P = -A*Re[(x_s - x + i eps)^(-a)]
#   alpha_true: pole order from dominant balance (1/(p-1) for f ~ c u^p)
#   inits     : multi-start initial guesses for x_s (span both sides of the
#               true pole; for control they straddle the virtual pole)
#   sign_guard: equilibrium-side consistency mask for K. For autonomous f
#               with f(0)=0, uniqueness forbids the flow through (x0,u0) to
#               cross the equilibrium 0, so fit values with sign(u) != sign(u0)
#               are provably NOT on the flow and their phase time is
#               meaningless (the integral passes through a zero of f and
#               diverges) -- their votes must be masked. tan has no
#               equilibrium (f = 1+u^2 > 0): guard off.

def _tau_u2(u):
    """tau(u) = 1/u  (f = u^2). Sign-preserving magnitude floor guards
    transient u ~ 0 fits; the redescending Tukey rho in the functional
    zeroes far-outlier votes."""
    mag = torch.clamp(u.abs(), min=0.05)
    return torch.sign(u) / mag


BENCHMARKS = {
    "tan": dict(
        rhs=lambda u: u * u + 1.0,
        exact=lambda x: np.tan(x),
        H=lambda u: (math.pi / 2 - torch.atan(u)),          # tau = pi/2 - atan(u)
        x_bc=0.0, u_bc=0.0,
        window=(0.0, 2.0), search=(-0.5, 2.5),
        xs_true=math.pi / 2, A_true=-1.0, alpha_true=1.0,
        inits=(0.5, 0.7, 1.3, 1.9), sign_guard=False,
        title=r"$u'=u^2+1,\ u(0)=0,\ \ u=\tan(x)$",
    ),
    "blowup": dict(
        rhs=lambda u: u * u,
        exact=lambda x: 1.0 / (1.0 - x),
        H=_tau_u2,                                           # tau = 1/u
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 2.0), search=(0.0, 2.0),
        xs_true=1.0, A_true=-1.0, alpha_true=1.0,
        inits=(0.5, 0.7, 1.3, 1.5), sign_guard=True,
        title=r"$u'=u^2,\ u(0)=1,\ \ u=1/(1-x)$",
    ),
    "control": dict(
        rhs=lambda u: u * u,
        exact=lambda x: -1.0 / (1.0 + x),
        H=_tau_u2,                                           # tau = 1/u
        x_bc=0.0, u_bc=-1.0,
        window=(0.0, 2.0), search=(-1.8, 2.0),
        xs_true=-1.0, A_true=-1.0, alpha_true=1.0,
        inits=(-1.5, -0.5, 0.5, 1.3), sign_guard=True,
        title=r"$u'=u^2,\ u(0)=-1,\ \ u=-1/(1+x)$ (virtual pole at $x_s=-1$)",
    ),
    "cubic": dict(
        rhs=lambda u: u ** 3,
        exact=lambda x: (1.0 - 2.0 * x) ** (-0.5),
        H=lambda u: 0.5 / torch.clamp(u * u, min=0.01),      # tau = 1/(2u^2)
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 0.5), search=(-0.3, 1.0),
        xs_true=0.5, A_true=-math.sqrt(0.5), alpha_true=0.5,
        inits=(0.1, 0.3, 0.7, 0.9), sign_guard=True,
        title=r"$u'=u^3,\ u(0)=1,\ \ u=(1-2x)^{-1/2}$ (edge pole, $\alpha=1/2$)",
    ),
    "u4": dict(
        rhs=lambda u: u ** 4,
        exact=lambda x: (1.0 - 3.0 * x) ** (-1.0 / 3.0),
        # tau = 1 / (3 u^3)  (f = u^4)
        H=lambda u: 1.0 / (3.0 * torch.clamp(u.abs() ** 3, min=1e-4)),
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 0.30), search=(-0.2, 0.6),
        xs_true=1.0 / 3.0, A_true=-(1.0 / 3.0) ** (1.0 / 3.0), alpha_true=1.0 / 3.0,
        inits=(0.05, 0.15, 0.4, 0.55), sign_guard=True,
        title=r"$u'=u^4,\ u(0)=1,\ \ u=(1-3x)^{-1/3}$ ($\alpha=1/3$)",
    ),
    "p32": dict(
        rhs=lambda u: u ** 1.5,
        exact=lambda x: 4.0 / (2.0 - x) ** 2,
        # tau = 2 / sqrt(u)  (f = u^(3/2))
        H=lambda u: 2.0 / torch.sqrt(torch.clamp(u, min=1e-3)),
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 1.5), search=(0.0, 2.5),
        xs_true=2.0, A_true=-4.0, alpha_true=2.0,
        inits=(0.5, 1.0, 1.7, 2.3), sign_guard=True,
        title=r"$u'=u^{3/2},\ u(0)=1,\ \ u=4/(2-x)^2$ ($\alpha=2$)",
    ),
    "u2_2u": dict(
        rhs=lambda u: u * u + 2.0 * u,
        exact=lambda x: 2.0 * np.exp(2.0 * x) / (3.0 - np.exp(2.0 * x)),
        # tau = (1/2) * ln((u+2)/u)  (f = u^2 + 2u = u(u+2))
        H=lambda u: 0.5 * torch.log((torch.clamp(u, min=1e-3) + 2.0)
                                     / torch.clamp(u, min=1e-3)),
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 0.50), search=(0.0, 0.9),
        xs_true=0.5 * math.log(3.0), A_true=-1.0, alpha_true=1.0,
        inits=(0.1, 0.25, 0.6, 0.8), sign_guard=True,
        title=r"$u'=u^2+2u,\ u(0)=1,\ \ u=2e^{2x}/(3-e^{2x})$ (linear+quadratic, $\alpha=1$)",
    ),
    "u3_u": dict(
        rhs=lambda u: u ** 3 + u,
        exact=lambda x: np.exp(x) / np.sqrt(2.0 - np.exp(2.0 * x)),
        # tau = (1/2) * ln((u^2+1)/u^2)  (f = u^3 + u = u(u^2+1))
        H=lambda u: 0.5 * torch.log((torch.clamp(u, min=1e-3) ** 2 + 1.0)
                                     / torch.clamp(u, min=1e-3) ** 2),
        x_bc=0.0, u_bc=1.0,
        window=(0.0, 0.30), search=(-0.2, 0.6),
        xs_true=0.5 * math.log(2.0), A_true=-math.sqrt(0.5), alpha_true=0.5,
        inits=(0.05, 0.15, 0.4, 0.55), sign_guard=True,
        title=r"$u'=u^3+u,\ u(0)=1,\ \ u=e^x/\sqrt{2-e^{2x}}$ (linear+cubic, $\alpha=1/2$)",
    ),
}


# ================= Collocation sampling with exclusion ball =================
def sample_collocation(xs_val, ball, window, n_total, near_frac=0.6):
    """
    Log-dense on both sides of the CURRENT pole estimate + uniform background.
    Rejects points within `ball` of xs_val (ball <= 0 disables rejection).
    near_frac = 0 gives pure uniform sampling (discovery phase). Runs on device.
    """
    w_lo, w_hi = window
    n_near = int(n_total * near_frac)
    n_uni = n_total - n_near
    span = w_hi - w_lo
    ball_eff = max(ball, 1e-9)

    def take(n, propose):
        chunks, got = [], 0
        while got < n:
            m = max(int(1.6 * (n - got)), 64)
            cand = propose(m)
            chunks.append(cand)
            got += cand.numel()
        return torch.cat(chunks)[:n]

    def propose_near(m):
        # distance log-uniform in [ball, span], random side
        d = torch.exp(torch.empty(m, device=device).uniform_(math.log(ball_eff), math.log(span)))
        side = torch.where(torch.rand(m, device=device) < 0.5, -1.0, 1.0)
        x = xs_val + side * d
        return x[(x >= w_lo) & (x <= w_hi)]

    def propose_uni(m):
        x = torch.empty(m, device=device).uniform_(w_lo, w_hi)
        return x[(x - xs_val).abs() >= ball_eff]

    parts = []
    if n_near > 0:
        parts.append(take(n_near, propose_near))
    if n_uni > 0:
        parts.append(take(n_uni, propose_uni))
    x = torch.cat(parts)
    return x[torch.randperm(x.numel(), device=device)].view(-1, 1)


# ================= Physics residual (dispatched per model type) =================
def physics_residual(model, x_col, rhs):
    """Returns (residual, u, ds): u feeds the phase-integral discovery
    functional, ds is the smooth-branch derivative for the roughness
    penalty (None for vanilla)."""
    if isinstance(model, LaurentPINN):
        u, du, ds = model.get_u_and_du(x_col)
        return du - rhs(u), u, ds
    u = model(x_col)
    du = torch.autograd.grad(u.sum(), x_col, create_graph=True, retain_graph=True)[0]
    return du - rhs(u), u, None


def pole_cap(xs_now, w_lo, w_hi, margin):
    """SIDE-AWARE window cap: collocation never CROSSES the current pole
    estimate. A pole strictly inside/right of the window stops the window
    short of it; a (virtual) pole at or left of the window start leaves the
    FULL window available -- the flow is regular everywhere on it (control)."""
    return xs_now - margin if xs_now > w_lo else w_hi


# ================= Training loop (two-phase: discovery -> refinement) =================
def train(model, bench, n_collocation=1000, n_epochs=6000, lr=1e-3, lam=1.0,
          clip=0.08, eps0=0.12, eps_min=1e-3, ball_factor=3.0, ball_min=0.05,
          near_frac=0.6, disc_frac=0.4, rough_w0=0.01, pole_k0=0.5,
          freeze_a_disc=True, cont_start=0.30, cont_growth=1.2,
          cont_margin=0.05, stage_tol=2e-3, fit_frac=0.75, r2a_frac=1.0,
          newton_dmax=0.05, use_newton=True, use_huber=False, side_aware=True,
          log_every=1000, tag=""):
    """
    Phase D (discovery, epochs [0, disc_frac*N)):
        Convergence-gated window continuation on [x_0, T_k]:
        * ADAPTIVE CAP: the window never CROSSES the current pole estimate
          (T <= x_s_hat - margin_eff when the pole sits inside/right of the
          window) because past the true pole the smooth net can no longer
          follow the flow (u -> +inf), the fit sags and the K votes are
          dragged off target (observed overshoot 0.99 -> 1.28). A virtual
          pole at/left of the window start caps nothing: the whole window
          is true flow (control benchmark).
        * margin_eff = max(cont_margin, 1.5*eps0 + 0.02) is EPS-AWARE: it
          keeps collocation out of the soft pole's steep zone |d| < eps
          (|dP/dx| ~ |A|/eps^2), where the smooth branch needs huge
          compensating slopes and the degrading fit poisons the K votes.
        * ERROR GATE: the window advances to the next geometric stage
          T_k = cont_start * cont_growth^k only after the fit on the current
          window has converged (EMA of physics loss < stage_tol, at least 30
          epochs per stage), or the per-stage deadline passes.
        eps pinned at eps0; pure uniform collocation; K weight pole_k0;
        residue A FROZEN at the class value A* = -1.
        x_s UPDATE = damped-Newton LOCATION step on the (masked) Tukey
        objective of the phase votes, NOT gradient descent. By the Osgood
        identity every on-flow collocation point votes v_i = x_i + tau(u_i)
        = x_s, so discovery is a robust LOCATION M-estimate of the votes;
        one Newton step per epoch converges in O(distance/newton_dmax)
        epochs. (Plain Adam on K was observed to crawl at ~1.5e-4/epoch --
        its normalized step is suppressed by vote scatter, E[g]/RMS[g] ~
        0.2 -- losing the race against the wrong-branch physics pull.)
        z is NOT differentiable, so physics/roughness/BC can never steer
        the pole: at the class residue the pole term ITSELF solves u' = f(u),
        and a pole misplaced inside the window hands physics a free
        wrong-branch flow whose IC-violation layer it softens by dragging
        x_s AWAY from the target (observed runaway 0.8 -> +1.54 on control
        instead of the virtual pole -1).
    Phase R1 (refinement, epochs [disc_frac*N, fit_frac*N)):
        SAME PRE-POLE WINDOW (T = x_s_hat - margin_eff, no continuation
        stages), cosine-anneal eps -> eps_min, uniform collocation + safety
        ball, A still frozen, K still active. Every collocation point sits
        on the true flow, so all K votes remain TRUE votes and K stays a
        consistent estimator of the pole position -- this is where the last
        digits of x_s are polished. (On a FULL window the post-pole side is
        fitted to a DISPLACED flow that satisfies the identity at ITS OWN
        pole, so its votes would self-confirm a misestimate -- hence K is
        confined to windows where every point sits on the true flow.)
    Phase R2 (final fit, remaining epochs): SIDE-AWARE commit-then-polish.
        Whether x_s keeps relocating in R2 is decided by where the pole sits
        relative to the PHYSICAL window:
        * IN-WINDOW pole (real blow-up inside the domain): Newton keeps
          relocating through the whole R2 by default (r2a_frac = 1.0) on the
          now-exact pure-physics votes (z non-differentiable so physics can
          never backslide it, K OFF, and the redescending influence zeroes
          post-pole continuation votes such as tan's far side 3*pi/2).
          Log-dense near-pole collocation points anchor the position, so
          relocation is a slow COUPLED CRAWL that unwinds the D/R1
          compensation bias as physics refits the pole shape -- committing
          early costs accuracy (tan: a 30%-commit leaves xs_err 6.1e-3 vs
          1.2e-4..3.1e-4 with the full crawl).
        * OFF-WINDOW (virtual) pole (control's analytic continuation):
          COMMIT (freeze) at R2 entry. No collocation point can approach it,
          R1's full-window votes have already extracted all location
          information, and every R2 relocation is pure jitter around it: a
          moving pole is a moving target the smooth branch keeps chasing
          (control plateaued at mae ~ 3e-2 across ALL starts while xs_err
          looked healthy; committed: mae 1.4e-5).
        r2a_frac < 1 forces an early commit for in-window poles (ablation);
        side_aware = False applies the commit fraction side-blind to every
        pole (pre-side-aware ablation; with r2a_frac = 1.0 it reproduces the
        v2 moving-target configuration).
        Throughout R2: FULL window, A unfrozen (residue recovery from pure
        physics: residual ~ -(A+1)/d^2 at near-pole points), log-dense
        sampling with exclusion ball, K OFF (no network-side vote
        alignment: a DIFFERENTIABLE z coupled to physics backslid on
        control 0.66 -> 0.88 in 300 epochs and worsened blowup
        0.9980 -> 0.9938).
    The roughness penalty is CONSTANT (rough_w0) in all phases: the true
    remainder after subtracting the true pole is smooth (identically zero
    for pure blow-up), so the penalty only bites spiky absorbers that
    would hide a misplaced pole.
    rough_w0 = 0 / pole_k0 = 0 disable the respective penalty (ablations).
    disc_frac = 0 + cont_start >= window end disables continuation (ablation).
    """
    w_lo, w_hi = bench["window"]
    x_bc = to_tensor(np.array([[bench["x_bc"]]]))
    u_bc = to_tensor(np.array([[bench["u_bc"]]]))
    rhs = bench["rhs"]

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    is_lau = isinstance(model, LaurentPINN)
    n_disc = int(round(disc_frac * n_epochs)) if is_lau else 0
    n_fit = max(int(round(fit_frac * n_epochs)), n_disc + 1) if is_lau else 0
    freeze_a = is_lau and model.learn_A and freeze_a_disc
    t_start = min(max(cont_start, w_lo + 1e-3), w_hi)
    n_stages = max(1, math.ceil(math.log(w_hi / t_start) / math.log(cont_growth))) if is_lau else 1
    # SMALL floors so the window can duck BELOW an interior pole estimate
    # (control transient): a window forced past the pole rides the displaced
    # flow, whose zero-votes dilute the K signal. Never binds for benchmarks
    # whose pole stays right of the window start.
    cont_floor = w_lo + 0.05 * (w_hi - w_lo)
    floor_r1 = w_lo + 0.02 * (w_hi - w_lo)
    # eps-aware cap margin: keeps Phase-D collocation out of the soft pole's
    # steep zone |d| < eps where the smooth branch needs huge compensating
    # slopes and the degrading fit poisons the K votes
    margin_eff = max(cont_margin, 1.5 * eps0 + 0.02) if is_lau else cont_margin
    # error-gated continuation state: advance the window only after the fit on
    # the current window converges, or the per-stage deadline passes
    stage_now, stage_epoch, phys_ema = 0, 0, None
    max_stage_ep = max(30, int(n_disc / max(n_stages, 1)))

    hist = {"total": [], "phys": [], "bc": [], "rough": [], "pole": [],
            "xs": [], "A": [], "eps": []}
    nan_epoch = -1

    for epoch in range(n_epochs):
        w_rough = 0.0
        w_pole = 0.0
        T_cur = w_hi

        # --- three-phase schedules: window, pole height, penalties, sampler
        if is_lau:
            in_disc = epoch < n_disc
            in_ref1 = (not in_disc) and (epoch < n_fit)
            tR1 = min((epoch - n_disc) / max(n_fit - n_disc, 1), 1.0)
            tR2 = (epoch - n_fit) / max(n_epochs - n_fit, 1)
            if freeze_a:
                # residue stays at the class value A* = -1 through D and R1;
                # only the final full-window fit (R2) recovers it from physics
                model.A.requires_grad_(not (in_disc or in_ref1))
            if model.soft_pole:
                if in_disc:
                    model.set_eps(eps0)
                elif in_ref1:
                    # cosine descent of the pole onto the real axis
                    model.set_eps(eps_min + (eps0 - eps_min) * 0.5 * (1.0 + math.cos(math.pi * tR1)))
                else:
                    model.set_eps(eps_min)
            # roughness is CONSTANT everywhere: the true Laurent remainder is
            # smooth (identically zero for pure blow-up), so the penalty only
            # bites spiky absorbers that would hide a misplaced pole
            w_rough = rough_w0
            if in_disc:
                w_pole = pole_k0
                near_eff = 0.0
                # soft pole is bounded everywhere -> no ball needed; hard pole
                # must always keep a safety ball
                ball = 0.0 if model.soft_pole else max(ball_factor * model.eps, ball_min)
                # error gate: extend the window only when the fit on the
                # current window has converged (>=30 epochs into the stage,
                # physics-loss EMA below stage_tol), or the deadline hits --
                # never open the window onto a still-sagging fit
                if stage_now < n_stages - 1:
                    conv = phys_ema is not None and phys_ema < stage_tol
                    if ((epoch - stage_epoch) >= max_stage_ep
                            or (conv and (epoch - stage_epoch) >= 30)):
                        stage_now += 1
                        stage_epoch = epoch
                # geometric stage target, ADAPTIVELY CAPPED so the window
                # never crosses the current pole estimate (right-side pole:
                # stop short of it; virtual pole at/left of the window
                # start: no cap, the whole window is true flow): the smooth
                # net always fits a representable flow and the K signal
                # stays unpolluted by the post-pole sag
                T_geo = t_start * (cont_growth ** stage_now)
                T_cap = pole_cap(float(model.xs.item()), w_lo, w_hi, margin_eff)
                T_cur = max(min(w_hi, T_geo, T_cap), cont_floor)
            elif in_ref1:
                if epoch == n_disc:
                    print(f"[{tag}] --- Phase D -> R1: keep pre-pole window, anneal eps, uniform sampling, K active, A frozen ---")
                # R1 polishes the position on a TRUE-FLOW window: every
                # collocation point sits on the exact flow (right-side pole:
                # pre-pole window; virtual pole at/left of the start: full
                # window), so all K votes are true votes and K remains a
                # consistent estimator (a displaced post-pole flow would
                # satisfy the identity at ITS OWN pole and self-confirm a
                # misestimate)
                w_pole = pole_k0
                near_eff = 0.0
                ball = max(ball_factor * model.eps, ball_min)
                T_cur = max(min(w_hi, pole_cap(float(model.xs.item()),
                                               w_lo, w_hi, margin_eff)), floor_r1)
            else:
                if epoch == n_fit:
                    print(f"[{tag}] --- Phase R1 -> R2: full window, unfreeze A, log-dense sampling + exclusion ball, K OFF; "
                          f"side-aware commit: in-window pole relocates through R2 (r2a_frac={r2a_frac:g}), off-window pole commits at entry ---")
                # final full-window fit: the (already accurate) pole explains
                # the blow-up region; the residue is recovered from pure
                # physics (residual ~ -(A+1)/d^2 at near-pole points)
                w_pole = 0.0
                near_eff = near_frac
                ball = max(ball_factor * model.eps, ball_min)
                T_cur = w_hi

        # --- collocation resampling on the current window
        if is_lau:
            x_col = sample_collocation(float(model.xs.item()), ball, (w_lo, T_cur),
                                        n_collocation, near_eff)
        else:
            x_col = torch.empty(n_collocation, 1, device=device).uniform_(w_lo, w_hi)
        x_col.requires_grad_(True)

        # --- losses
        res, u, ds = physics_residual(model, x_col, rhs)
        l_phys = torch.mean(res ** 2)
        if is_lau:
            l_rough = torch.mean(ds ** 2)
            # phase-integral votes (GENERAL, Osgood-based): every point of
            # any exact flow obeys v_i = x_i + tau(u_i) = x_s. tau is built
            # from the KNOWN f alone -- no residue, no pole order, no
            # solution shape assumed (the old Riccati moment is exactly the
            # f = u^2 case: tau = 1/u <=> u*(x - x_s) + 1 = 0).
            tau = bench["H"](u)
            votes = x_col + tau
            if bench.get("sign_guard", False) and bench["u_bc"] != 0.0:
                # EQUILIBRIUM-SIDE CONSISTENCY: for autonomous f with f(0)=0,
                # uniqueness forbids the flow through (x0, u0) to cross 0, so
                # fit values with sign(u) != sign(u0) are provably off-flow
                # and their phase time is meaningless -- exclude their votes.
                # Without the mask, a misplaced pole's positive transient
                # (u > 0 while the true flow is negative) generates positive
                # votes that snowball x_s AWAY from the target (observed on
                # control: x_s drifted 0.8 -> +1.56 instead of -1).
                s0 = math.copysign(1.0, bench["u_bc"])
                m = (torch.sign(u) == s0).to(votes.dtype)
            else:
                m = torch.ones_like(votes)
            xs_cur = float(model.xs.item())
            w_res = votes - xs_cur                # current vote residuals
            av = w_res.abs()
            w2 = w_res * w_res
            # Influence-family choice (ablation switch). Tukey biweight
            # (c = 1) is the default: rho'(w) = 0 beyond |w| = 1, so votes
            # of DISCONNECTED branches -- e.g. the analytic continuation of
            # tan past its pole votes exactly 3*pi/2 -- get ZERO influence
            # instead of Huber's clamped +-1, whose residual mass drags the
            # location estimate (observed: tan stalls at pi/2 + 8e-3 from
            # every start). z is not differentiable, so K never moves x_s;
            # the Newton step below is the actual updater.
            if use_huber:
                rho = torch.where(av < 1.0, 0.5 * w2, av - 0.5)
            else:
                rho = torch.where(av < 1.0, (1.0 - (1.0 - w2) ** 3) / 6.0,
                                  torch.full_like(w_res, 1.0 / 6.0))
            l_pole = (rho * m).sum() / m.sum().clamp(min=1.0)
        else:
            l_rough = torch.zeros((), device=device)
            l_pole = torch.zeros((), device=device)
        l_bc = torch.mean((model(x_bc) - u_bc) ** 2)
        loss = l_bc + lam * l_phys
        if w_rough > 0.0:
            loss = loss + w_rough * l_rough
        if w_pole > 0.0:
            loss = loss + w_pole * l_pole

        if not torch.isfinite(loss):
            nan_epoch = epoch
            print(f"[{tag}] loss became non-finite at epoch {epoch}; stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        if clip and clip > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip)
        optimizer.step()
        scheduler.step()

        # --- damped-Newton LOCATION step on the phase votes: the ONLY
        # updater of x_s. Active in all of D/R1 (tR2 < 0). In R2 it is
        # SIDE-AWARE (commit-then-polish): an IN-window pole keeps
        # relocating while tR2 < r2a_frac (default 1.0 = the whole R2) --
        # log-dense near-pole collocation points anchor it, so the
        # relocation is a slow coupled crawl that unwinds the D/R1
        # compensation bias as physics refits the pole shape (tan: a
        # 30%-commit leaves xs_err 6.1e-3 vs 1.2e-4..3.1e-4 with the full
        # crawl); an OFF-window (virtual) pole COMMITS at R2 entry -- no
        # collocation point can approach it, R1's full-window votes have
        # already extracted all location information, and every R2
        # relocation is pure jitter around it: a moving pole is a moving
        # target the smooth branch keeps chasing (control plateaued at mae
        # ~3e-2 across all starts with xs_err looking healthy, and doubling
        # the budget did not cure it; committed: mae 1.4e-5).
        # Discovery is a Tukey-biweight location M-estimate: solve
        # mean[psi(v - x_s)] = 0 over the masked votes with
        # psi(w) = w (1 - w^2)^2 for |w| < 1 and 0 beyond, by one damped
        # quasi-Newton step per epoch,
        #   delta = mean[psi(w)] / frac(|w| < 1),
        # falling back to a full damped step toward the masked vote MEDIAN
        # when almost no vote is an inlier (sign-of-mean is scatter-blind;
        # the median is not). Converges in O(distance / newton_dmax)
        # epochs, immune to the vote-scatter suppression that limits Adam
        # (~1.5e-4 per epoch observed).
        if is_lau and model.learn_xs and use_newton:
            if side_aware:
                span_w = w_hi - w_lo
                in_window = ((xs_cur - w_lo) >= -0.05 * span_w
                             and (w_hi - xs_cur) >= -0.05 * span_w)
                active = tR2 < 0 or (in_window and tR2 < r2a_frac)
            else:
                # pre-side-aware ablation: the commit fraction applies
                # side-blind to every pole (r2a_frac=1.0 == v2
                # moving-target configuration)
                active = tR2 < r2a_frac
            if active:
                with torch.no_grad():
                    den = m.sum().clamp(min=1.0)
                if use_huber:
                    # Huber: psi(w) = clip(w, +-1) -- influence never
                    # redescends, far-branch votes keep a +-1 pull
                    inl = m
                    psi = torch.clamp(w_res, -1.0, 1.0)
                else:
                    # Tukey biweight: psi(w) = w (1 - w^2)^2 inside |w|<1,
                    # ZERO beyond -- far-branch votes get no pull
                    inl = (av < 1.0).to(w_res.dtype) * m
                    psi = w_res * (1.0 - w2) ** 2
                num = (psi * inl).sum() / den
                quad = inl.sum() / den
                if float(quad) > 0.15:
                    delta = num / quad
                elif float(m.sum()) > 0.0:
                    med = votes.masked_select(m > 0.0).median()
                    delta = torch.sign(med - xs_cur) * newton_dmax
                else:
                    delta = torch.zeros((), device=votes.device)
                delta = float(torch.clamp(delta, -newton_dmax, newton_dmax))
                model.set_xs(xs_cur + delta)

        hist["total"].append(loss.item())
        pv = l_phys.item()
        hist["phys"].append(pv)
        phys_ema = pv if phys_ema is None else 0.9 * phys_ema + 0.1 * pv
        hist["bc"].append(l_bc.item())
        hist["rough"].append(l_rough.item() if is_lau else 0.0)
        hist["pole"].append(l_pole.item() if is_lau else 0.0)
        if is_lau:
            hist["xs"].append(float(model.xs.item()))
            hist["A"].append(float(model.A.item()))
            hist["eps"].append(model.eps)

        if (epoch + 1) % log_every == 0:
            msg = f"[{tag}] Epoch {epoch + 1:5d} | Loss {loss.item():.2e}"
            if is_lau:
                msg += (f" | T {T_cur:.2f} | xs {hist['xs'][-1]:.4f} | A {hist['A'][-1]:+.4f}"
                        f" | eps {model.eps:.4f} | rough {hist['rough'][-1]:.2e}"
                        f" | K {hist['pole'][-1]:.2e} | w [{w_rough:.2f},{w_pole:.2f}]")
            print(msg)

    hist["nan_epoch"] = nan_epoch
    return hist


# ================= Evaluation =================
def evaluate(model, bench, ball_eval=0.07, n_grid=1200):
    """Errors on a grid that excludes a ball around the TRUE pole (and around
    the model pole in hard-pole mode, where forward evaluation would be inf)."""
    w_lo, w_hi = bench["window"]
    g = np.linspace(w_lo, w_hi, n_grid * 3)
    keep = np.ones_like(g, dtype=bool)
    if bench.get("xs_true") is not None:
        keep &= np.abs(g - bench["xs_true"]) >= ball_eval
    if isinstance(model, LaurentPINN) and model.eps == 0.0:
        keep &= np.abs(g - float(model.xs.item())) >= ball_eval
    g = g[keep]

    x_t = to_tensor(g.reshape(-1, 1))
    model.eval()
    with torch.no_grad():
        u_pred = model(x_t).cpu().numpy().ravel()
    u_exact = bench["exact"](g)
    err = np.abs(u_pred - u_exact)

    m = {"mae": float(err.mean()), "max_err": float(err.max())}
    if isinstance(model, LaurentPINN):
        m["xs_hat"] = float(model.xs.item())
        m["A_hat"] = float(model.A.item())
        if bench.get("xs_true") is not None:
            m["xs_err"] = abs(m["xs_hat"] - bench["xs_true"])
            m["A_err"] = abs(m["A_hat"] - bench["A_true"])
    return g, u_pred, u_exact, m


def compute_residual(model, bench, g):
    x_t = to_tensor(g.reshape(-1, 1), requires_grad=True)
    if isinstance(model, LaurentPINN):
        u, du, _ = model.get_u_and_du(x_t)
    else:
        u = model(x_t)
        du = torch.autograd.grad(u.sum(), x_t, create_graph=False)[0]
    return (du - bench["rhs"](u)).detach().abs().cpu().numpy().ravel()


# ================= Visualization =================
def plot_results(model, bench, hist, g, u_pred, u_exact, metrics, save_fig=None, y_lim=None):
    res = compute_residual(model, bench, g)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{bench['title']}  —  {'Laurent PINN (learnable pole)' if isinstance(model, LaurentPINN) else 'Vanilla PINN'}",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    # The evaluation grid excludes a ball around the pole (discontinuous),
    # but ax.plot connects consecutive points by default, producing a spurious
    # line across the asymptote. Insert NaN at the jump so the curve breaks
    # properly at the singularity.
    def _break_at_jumps(x, y):
        if len(x) < 2:
            return x, y
        dx = np.diff(x)
        jumps = np.where(dx > 5 * np.median(dx))[0]
        if len(jumps) == 0:
            return x, y
        # insert NaN right after each jump point to break the line
        x_list, y_list = list(x), list(y)
        for j in reversed(jumps):
            x_list.insert(j + 1, float(x[j + 1]))
            y_list.insert(j + 1, float("nan"))
        return np.array(x_list), np.array(y_list)

    g_broken, u_exact_broken = _break_at_jumps(g, u_exact)
    _, u_pred_broken = _break_at_jumps(g, u_pred)
    ax.plot(g_broken, u_exact_broken, "k-", lw=2, label="Exact")
    ax.plot(g_broken, u_pred_broken, "r--", lw=1.8, label="PINN")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Solution vs exact")
    ax.legend(); ax.grid(alpha=0.3)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        # auto: clamp extreme values so the characteristic S-shape of
        # tan-like blowup is visible (raw pole-adjacent values ~15 flatten
        # everything into a V)
        med = np.median(np.abs(u_exact))
        cap = max(med * 8, 5.0)
        ax.set_ylim(-cap, cap)
    # draw the vertical asymptote at the true pole (if known) to make the
    # blow-up structure visually obvious — the exclusion ball removes the
    # pole-adjacent points, so without this line the singularity looks like
    # a V-shaped kink instead of an asymptote
    if bench.get("xs_true") is not None and isinstance(model, LaurentPINN):
        ylo, yhi = ax.get_ylim()
        ax.axvline(bench["xs_true"], color="gray", ls=":", lw=1.2, alpha=0.7,
                   label=r"$x_s^*$ (asymptote)")
        # also draw the learned pole as a fainter line
        ax.axvline(float(model.xs.item()), color="tab:red", ls="--", lw=1.0,
                   alpha=0.5, label=r"$\hat x_s$ (learned)")
        ax.legend()

    ax = axes[0, 1]
    for k, lab in [("total", "total"), ("phys", "physics"), ("bc", "bc")]:
        ax.semilogy(hist[k], label=lab, lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(g, res, "b-", lw=1.5)
    ax.fill_between(g, res, alpha=0.2, color="blue")
    ax.set_xlabel("x"); ax.set_ylabel(r"$|du/dx - f(u)|$")
    ax.set_title("Physics residual over domain"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if isinstance(model, LaurentPINN) and bench.get("xs_true") is not None:
        ep = np.arange(len(hist["xs"]))
        ax.plot(ep, hist["xs"], lw=1.8, color="tab:blue", label=r"$x_s$ (learned)")
        ax.axhline(bench["xs_true"], ls="--", color="k", lw=1.5, label=r"$x_s^*$ (true)")
        ax2 = ax.twinx()
        ax2.plot(ep, hist["A"], lw=1.5, color="tab:red", alpha=0.8, label="A (learned)")
        ax2.axhline(bench["A_true"], ls=":", color="tab:red", lw=1.2)
        ax2.set_ylabel("A", color="tab:red")
        ax.set_xlabel("Epoch"); ax.set_ylabel(r"$x_s$", color="tab:blue")
        ax.set_title("Singularity discovery trajectory")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right")
        ax.grid(alpha=0.3)
    else:
        ax.semilogy(g, np.abs(u_pred - u_exact), "g-", lw=1.5)
        ax.fill_between(g, np.abs(u_pred - u_exact), alpha=0.2, color="green")
        ax.set_xlabel("x"); ax.set_ylabel(r"$|u_{PINN}-u_{exact}|$")
        ax.set_title("Pointwise absolute error"); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig)), exist_ok=True)
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"figure saved -> {save_fig}")
    else:
        plt.show()


# ================= Multi-start robustness suite =================
def run_multistart(bench_name, inits=None, epochs=6000, save_fig=None):
    bench = BENCHMARKS[bench_name]
    if inits is None:
        inits = bench.get("inits", (0.5, 0.7, 1.3, 1.5))
    hists, finals = [], []
    for xs0 in inits:
        seed_everything(1314)
        model = LaurentPINN(window=bench["window"], search=bench["search"],
                            xs_init=xs0, A_init=bench["A_true"],
                            alpha=bench["alpha_true"], soft_pole=True,
                            x_bc=bench["x_bc"], u_bc=bench["u_bc"]).to(device)
        hist = train(model, bench, n_epochs=epochs, tag=f"init={xs0}")
        _, _, _, m = evaluate(model, bench)
        hists.append((xs0, hist))
        finals.append((xs0, m))
        print(f"[SUMMARY] tag=multistart init={xs0} xs_hat={m['xs_hat']:.6f} "
              f"xs_err={m.get('xs_err', float('nan')):.2e} A_hat={m['A_hat']:+.4f} "
              f"mae={m['mae']:.2e} max_err={m['max_err']:.2e} nan_epoch={hist['nan_epoch']}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    fig.suptitle(f"Multi-start singularity discovery — {bench['title']}", fontsize=13, fontweight="bold")
    for xs0, h in hists:
        axes[0].plot(h["xs"], lw=1.6, label=f"init {xs0}")
        axes[1].plot(h["A"], lw=1.6, label=f"init {xs0}")
    axes[0].axhline(bench["xs_true"], ls="--", color="k", lw=1.8, label=r"true $x_s^*$")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel(r"$x_s$")
    axes[0].set_title(r"$x_s$ trajectories"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(bench["A_true"], ls="--", color="k", lw=1.8, label=r"true $A^*$")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("A")
    axes[1].set_title("Residue trajectories"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(os.path.abspath(save_fig)), exist_ok=True)
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"figure saved -> {save_fig}")
    else:
        plt.show()
    return finals


# ================= CLI entry =================
def main():
    p = argparse.ArgumentParser(description="Laurent PINN with learnable singularity position")
    p.add_argument("--bench", choices=list(BENCHMARKS), default="blowup")
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--seed", type=int, default=1314)
    p.add_argument("--xs-init", type=float, default=0.8)
    p.add_argument("--a-init", type=float, default=None,
                   help="pole residue init; default = dominant-balance value A* of the class")
    p.add_argument("--fixed-xs", type=float, default=None, help="freeze pole at this position")
    p.add_argument("--no-learn-a", action="store_true")
    p.add_argument("--no-soft", action="store_true", help="disable pole annealing (hard pole)")
    p.add_argument("--vanilla", action="store_true", help="baseline vanilla PINN")
    p.add_argument("--clip", type=float, default=0.08, help="<=0 disables clipping")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--disc-frac", type=float, default=0.4,
                   help="fraction of epochs spent in the discovery phase D")
    p.add_argument("--fit-frac", type=float, default=0.75,
                   help="epoch fraction at which R1 ends and the full-window fit R2 starts")
    p.add_argument("--r2a-frac", type=float, default=1.0,
                   help="fraction of R2 before an IN-WINDOW x_s is COMMITTED (frozen) for the "
                        "static-target polish; 1.0 (default) = relocation through the whole R2 -- "
                        "near-pole collocation anchors in-window poles and tan needs the full "
                        "crawl (30%% commit: xs_err 6.1e-3 vs ~3e-4); an OFF-window (virtual) "
                        "pole commits at R2 entry regardless (side-aware)")
    p.add_argument("--no-side-aware", action="store_true",
                   help="apply the commit fraction side-blind to every pole (ablation; with "
                        "--r2a-frac 1.0 this is the v2 config: control moving-target plateau "
                        "mae ~3e-2)")
    p.add_argument("--rough-w0", type=float, default=0.01,
                   help="initial roughness weight; 0 disables the roughness penalty")
    p.add_argument("--pole-k0", type=float, default=0.5,
                   help="phase-integral discovery functional weight; 0 disables it (ablation)")
    p.add_argument("--no-freeze-a", action="store_true",
                   help="learn the residue during the discovery phase too (ablation)")
    p.add_argument("--no-newton", action="store_true",
                   help="freeze x_s entirely: no location step (ablation; with --pole-k0 0 this is full discovery off)")
    p.add_argument("--huber", action="store_true",
                   help="clamped (Huber) influence in K and the location step instead of the "
                        "redescending Tukey biweight (ablation: 3*pi/2 far-branch vote stall)")
    p.add_argument("--no-guard", action="store_true",
                   help="disable the equilibrium-side consistency mask (ablation: off-flow vote snowball)")
    p.add_argument("--cont-start", type=float, default=0.30,
                   help="initial continuation window length (>= window end disables)")
    p.add_argument("--cont-growth", type=float, default=1.2,
                   help="geometric growth factor of the continuation window")
    p.add_argument("--save-fig", default=None)
    p.add_argument("--ylim", type=float, nargs=2, default=None,
                   metavar=("YLO", "YHI"),
                   help="y-axis limits for the solution plot (e.g. --ylim -5 5); "
                        "default = auto-scaled to show the S-shape clearly")
    p.add_argument("--suite", choices=["multistart"], default=None)
    p.add_argument("--tag", default="run")
    args = p.parse_args()

    bench = BENCHMARKS[args.bench]
    if args.no_guard:
        bench = dict(bench, sign_guard=False)

    if args.suite == "multistart":
        run_multistart(args.bench, epochs=args.epochs, save_fig=args.save_fig)
        return

    seed_everything(args.seed)
    if args.vanilla:
        model = VanillaPINN(x_min=bench["window"][0], x_max=bench["window"][1],
                            x_bc=bench["x_bc"], u_bc=bench["u_bc"]).to(device)
        tag = f"{args.tag}|vanilla|{args.bench}"
    else:
        xs0 = args.fixed_xs if args.fixed_xs is not None else args.xs_init
        A0 = args.a_init if args.a_init is not None else bench["A_true"]
        model = LaurentPINN(window=bench["window"], search=bench["search"],
                            xs_init=xs0, A_init=A0, alpha=bench["alpha_true"],
                            learn_xs=args.fixed_xs is None, learn_A=not args.no_learn_a,
                            soft_pole=not args.no_soft,
                            x_bc=bench["x_bc"], u_bc=bench["u_bc"]).to(device)
        tag = f"{args.tag}|xs0={xs0}|{args.bench}" + ("|hard" if args.no_soft else "")

    hist = train(model, bench, n_epochs=args.epochs, lr=args.lr, clip=args.clip,
                 disc_frac=args.disc_frac, fit_frac=args.fit_frac,
                 r2a_frac=args.r2a_frac, rough_w0=args.rough_w0,
                 pole_k0=args.pole_k0, freeze_a_disc=not args.no_freeze_a,
                 cont_start=args.cont_start, cont_growth=args.cont_growth,
                 use_newton=not args.no_newton, use_huber=args.huber,
                 side_aware=not args.no_side_aware, tag=tag)
    g, u_pred, u_exact, m = evaluate(model, bench)

    extra = ""
    if "xs_hat" in m:
        extra = (f" xs_hat={m['xs_hat']:.6f} A_hat={m['A_hat']:+.4f}")
        if "xs_err" in m:
            extra += f" xs_err={m['xs_err']:.2e} A_err={m['A_err']:.2e}"
    print(f"[SUMMARY] tag={tag} mae={m['mae']:.2e} max_err={m['max_err']:.2e}"
          f"{extra} nan_epoch={hist['nan_epoch']}")

    plot_results(model, bench, hist, g, u_pred, u_exact, m,
                 save_fig=args.save_fig, y_lim=args.ylim)


if __name__ == "__main__":
    main()
