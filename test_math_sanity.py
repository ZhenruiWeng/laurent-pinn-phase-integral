"""Fast math sanity checks for the generalized Laurent pole + phase integral.

Run: python test_math_sanity.py
Checks:
  1. New general-alpha soft pole reduces EXACTLY to the old alpha=1
     Lorentzian p = A*d_old/(d_old^2 + eps^2) (d_old = x - x_s).
  2. Closed-form dP/dx matches central finite differences (all alphas, soft/hard).
  3. Phase-integral identity x + tau(u_exact(x)) = x_s holds pointwise on all
     four benchmark exact solutions.
  4. Hard-pole far side matches the alpha=1 limit A/(x - x_s).
  5. tan's post-pole analytic continuation votes EXACTLY 3*pi/2 and lies
     beyond the Tukey cutoff for every candidate pole in the search window
     (the analytical basis for the redescending influence function).
"""
import math
import numpy as np
import torch

from laurent_pinn_learnable_xs import BENCHMARKS, LaurentPINN

torch.set_default_dtype(torch.float64)
dev = "cpu"
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


# ---------- 1. alpha=1 soft pole == old Lorentzian ----------
xs_val, A_val, eps = 0.63, -0.7, 0.12
model = LaurentPINN(window=(0.0, 2.0), xs_init=xs_val, A_init=A_val,
                    alpha=1.0, soft_pole=True, x_bc=0.0, u_bc=0.0).to(dev).double()
model.set_eps(eps)
x = torch.linspace(-1.0, 2.5, 200, dtype=torch.float64).view(-1, 1)
d_new = model.xs.detach() - x
p_new, dp_new = model._pole(d_new)
d_old = x - model.xs.detach()
denom = d_old * d_old + eps ** 2
p_old = model.A.detach() * d_old / denom
dp_old = model.A.detach() * (eps ** 2 - d_old * d_old) / (denom * denom)
err_p = (p_new - p_old).abs().max().item()
err_dp = (dp_new - dp_old).abs().max().item()
check("soft pole alpha=1 vs old Lorentzian", err_p < 1e-12, f"max|dP|={err_p:.2e}")
check("soft dP/dx alpha=1 vs old Lorentzian", err_dp < 1e-12, f"max|ddP|={err_dp:.2e}")

# ---------- 2. closed-form dP/dx vs finite differences ----------
for alpha in (1.0, 0.5, 0.25, 1.5):
    m = LaurentPINN(window=(0.0, 2.0), xs_init=0.7, A_init=-0.9,
                   alpha=alpha, soft_pole=True, x_bc=0.0, u_bc=0.0).to(dev).double()
    m.set_eps(0.07)
    xg = torch.linspace(-1.5, 2.0, 400, dtype=torch.float64).view(-1, 1)
    d = m.xs.detach() - xg
    p, dp = m._pole(d)
    h = 1e-6
    dp_fd = (m._pole(d + h)[0] - m._pole(d - h)[0]) / (2 * h)
    # dP/dx = -dP/dd  (d = xs - x)
    err = (dp - (-dp_fd)).abs().max().item()
    check(f"closed-form dP/dx alpha={alpha}", err < 1e-5, f"max err={err:.2e}")

# hard pole far side, alpha=1: expect A/(x - x_s)
m1 = LaurentPINN(window=(0.0, 2.0), xs_init=0.7, A_init=-1.3, alpha=1.0,
                soft_pole=False, x_bc=0.0, u_bc=0.0).to(dev).double()
xf = torch.tensor([[1.9]], dtype=torch.float64)
p_f, _ = m1._pole(m1.xs.detach() - xf)
expect = float(m1.A.detach()) / (1.9 - float(m1.xs.detach()))
check("hard far side alpha=1 == A/(x-xs)",
      abs(p_f.item() - expect) < 1e-12, f"got {p_f.item():.6f} want {expect:.6f}")

# ---------- 3. phase-integral identity on exact solutions ----------
for name, b in BENCHMARKS.items():
    w_lo, w_hi = b["window"]
    g = np.linspace(w_lo + 0.02, min(w_hi, b["xs_true"]) - 0.02, 500)
    u_ex = torch.tensor(b["exact"](g), dtype=torch.float64).view(-1, 1)
    tau = b["H"](u_ex)
    votes = torch.tensor(g, dtype=torch.float64).view(-1, 1) + tau - b["xs_true"]
    err = votes.abs().max().item()
    check(f"identity x+tau(u)=xs  [{name}]", err < 1e-6,
          f"max|vote|={err:.2e} (xs*={b['xs_true']:.4f})")

# ---------- 4. dominant-balance consistency: residual of pure pole ansatz ----------
# u = P(x) alone must nearly solve the ODE near the pole for the class values
for name in ("blowup", "cubic"):
    b = BENCHMARKS[name]
    m = LaurentPINN(window=b["window"], search=b["search"], xs_init=b["xs_true"],
                   A_init=b["A_true"], alpha=b["alpha_true"], soft_pole=False,
                   x_bc=b["x_bc"], u_bc=b["u_bc"]).to(dev).double()
    # zero the smooth net's output layer -> u = P(x) exactly (the untrained
    # Xavier init otherwise injects an O(1) smooth-branch contaminant)
    with torch.no_grad():
        m.smooth_net[-1].weight.zero_()
        m.smooth_net[-1].bias.zero_()
    dists = np.geomspace(0.01, 0.2, 50)
    xx = torch.tensor(b["xs_true"] - dists, dtype=torch.float64).view(-1, 1)
    xx.requires_grad_(True)
    u, du, _ = m.get_u_and_du(xx)
    res = (du - b["rhs"](u)) / (u.abs() ** 3)  # relative to leading term
    check(f"pole ansatz dominant balance [{name}]",
          res.abs().max().item() < 0.2, f"max rel res={res.abs().max().item():.2e}")

# ---------- 5. disconnected-branch votes: tan continuation votes 3*pi/2 ----------
# For x > pi/2 the analytic continuation of tan obeys atan(tan x) = x - pi,
# so its phase votes are EXACTLY 3*pi/2 -- a fixed distance ~pi from any
# candidate pole. The redescending (Tukey) influence must reject them for
# EVERY candidate in the search window; Huber's clamp keeps a +-1 pull.
bt = BENCHMARKS["tan"]
g5 = np.linspace(bt["xs_true"] + 0.02, bt["window"][1] - 1e-3, 300)
u5 = torch.tensor(bt["exact"](g5), dtype=torch.float64).view(-1, 1)   # negative branch
votes5 = torch.tensor(g5, dtype=torch.float64).view(-1, 1) + bt["H"](u5)
err5 = (votes5 - 3 * math.pi / 2).abs().max().item()
check("tan continuation votes == 3*pi/2", err5 < 1e-8,
      f"max|vote-3pi/2|={err5:.2e}")
cand5 = torch.linspace(bt["search"][0], bt["search"][1], 400,
                       dtype=torch.float64).view(-1, 1)
mind5 = (votes5.view(1, -1) - cand5).abs().min().item()
check("continuation votes beyond Tukey cutoff for ALL candidates", mind5 > 1.0,
      f"min|vote-candidate|={mind5:.3f} > 1")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
