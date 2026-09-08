"""P6 — post-hoc predictive uncertainty that provably cannot move the mean.

The plan approves uncertainty as a **post-hoc module only**, on frozen base
checkpoints, and states the constraint plainly: *"The uncertainty module may not
change the registered mean prediction."*  It then names three tests:

    base checkpoint hash unchanged
    zero gradient into base model
    bit-identical mean prediction before / after attaching the head

Those are not ceremony.  A calibrator trained jointly with the mean model
quietly re-selects the mean — the registered number in the paper stops being the
number that was registered, and nobody sees it happen because the mean model's
loss still looks fine.  So the guarantee here is structural rather than
procedural:

* the base model is loaded, ``requires_grad_(False)``, and put in eval mode;
* the head consumes **detached** features only, so no gradient path to the base
  exists to begin with — `torch.autograd.grad` raises rather than returning
  zeros if one were ever introduced;
* ``forward`` returns the base model's own tensor object for the mean, not a
  recomputation, so bit-identity is guaranteed by construction and then
  asserted anyway.

What the head predicts
----------------------
A log-variance per channel, from stop-gradient features of the frozen model plus
the evidence the DFS estimator already computes.  Evidence is the natural input:
a query in a well-observed neighbourhood should be more certain, and omega is
exactly the quantity that says so.  Trained by Gaussian NLL, which is proper —
it cannot be improved by misreporting the spread.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def state_dict_hash(sd: dict) -> str:
    """Order-independent SHA-256 over a state dict's bytes.

    Used to assert the base checkpoint is unchanged after the head is trained.
    Sorting the keys matters: dict order is not part of the model.
    """
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        t = v.detach().cpu().contiguous() if torch.is_tensor(v) else torch.as_tensor(v)
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class UncertaintyHead(nn.Module):
    """Log-variance per channel from stop-gradient features."""

    def __init__(self, n_features: int, n_channels: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_channels))
        # start near unit variance so the first NLL steps are not dominated by
        # an arbitrary initial scale
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # clamp keeps NLL finite if the head is pushed to an extreme early on;
        # exp(-8)..exp(8) spans 3e-4..3e3 in variance, far wider than any real
        # calibration, so it never binds in practice
        return self.net(feats).clamp(-8.0, 8.0)


class CalibratedModel(nn.Module):
    """A frozen mean model plus a variance head that cannot reach it."""

    def __init__(self, base: nn.Module, head: UncertaintyHead):
        super().__init__()
        self.base = base
        self.head = head
        self.base.eval()
        self.base.requires_grad_(False)
        self.base_hash = state_dict_hash(self.base.state_dict())

    def train(self, mode: bool = True):
        """Keep the base in eval no matter what the trainer does.

        `Module.train()` recurses into children, so a plain `model.train()`
        would flip the frozen base's dropout and batch-norm back on and change
        its predictions -- silently, and only during training.
        """
        super().train(mode)
        self.base.eval()
        return self

    def features(self, s: dict, mean: torch.Tensor) -> torch.Tensor:
        """Stop-gradient features: the mean prediction and the query geometry.

        Everything is detached at the boundary.  The head therefore sees the
        base model's output as data, exactly as a downstream consumer would.
        """
        q = s["query"].to(mean.dtype)
        feats = [mean.detach(), q.detach()]
        om = getattr(self.base, "last_omega", None)
        if om is not None and torch.is_tensor(om) and om.numel():
            # one scalar summary of the evidence available to this sample,
            # broadcast to every query: a well-observed month should be more
            # certain everywhere than a sparse one
            tot = om.detach().double().sum().to(mean.dtype)
            feats.append(tot.expand(mean.shape[0], 1))
        else:
            feats.append(torch.zeros(mean.shape[0], 1, dtype=mean.dtype,
                                     device=mean.device))
        return torch.cat(feats, dim=-1)

    def forward(self, s: dict):
        with torch.no_grad():
            mean = self.base(s)
        return mean, self.head(self.features(s, mean))


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor,
                 target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Proper scoring rule: 0.5*(logvar + (y-mu)^2/var), masked."""
    var = logvar.exp()
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    m = mask.to(nll.dtype)
    return (nll * m).sum() / m.sum().clamp(min=1.0)


# --------------------------------------------------------------------------
# calibration diagnostics
# --------------------------------------------------------------------------
def crps_gaussian(mean: np.ndarray, sigma: np.ndarray,
                  y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting 2005).

    Analytic rather than sampled: a sampled CRPS adds Monte-Carlo noise to a
    number the cross-check compares between two tracks, and the closed form
    removes that source of disagreement entirely.
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mean) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                    - 1.0 / np.sqrt(np.pi))


def spread_skill(sigma: np.ndarray, err: np.ndarray, n_bins: int = 10) -> dict:
    """Binned spread-vs-skill. A calibrated model has RMSE(bin) ~ spread(bin)."""
    ok = np.isfinite(sigma) & np.isfinite(err)
    sigma, err = sigma[ok], err[ok]
    if sigma.size == 0:
        return {"bins": [], "slope": float("nan")}
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    bins = []
    for i in range(n_bins):
        m = (sigma >= edges[i]) & (sigma < edges[i + 1])
        if m.sum() < 2:
            continue
        bins.append({"spread": float(np.sqrt(np.mean(sigma[m] ** 2))),
                     "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                     "n": int(m.sum())})
    if len(bins) < 2:
        return {"bins": bins, "slope": float("nan")}
    x = np.array([b["spread"] for b in bins])
    y = np.array([b["rmse"] for b in bins])
    slope = float(np.polyfit(x, y, 1)[0])
    return {"bins": bins, "slope": slope,
            "note": "slope 1.0 is perfectly calibrated; <1 over-dispersed, "
                    ">1 under-dispersed"}


def interval_coverage(mean: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                      levels=(0.5, 0.8, 0.9, 0.95, 0.99)) -> dict:
    """Empirical coverage of central predictive intervals."""
    from scipy.stats import norm
    out = {}
    ok = np.isfinite(mean) & np.isfinite(sigma) & np.isfinite(y)
    m, s, t = mean[ok], np.maximum(sigma[ok], 1e-12), y[ok]
    for p in levels:
        z = norm.ppf(0.5 + p / 2)
        out[f"{p:.2f}"] = float(np.mean(np.abs(t - m) <= z * s))
    out["n"] = int(ok.sum())
    return out


def reliability(mean: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                n_bins: int = 10) -> dict:
    """PIT histogram. Uniform means calibrated; U-shaped means over-confident."""
    from scipy.stats import norm
    ok = np.isfinite(mean) & np.isfinite(sigma) & np.isfinite(y)
    pit = norm.cdf((y[ok] - mean[ok]) / np.maximum(sigma[ok], 1e-12))
    hist, edges = np.histogram(pit, bins=n_bins, range=(0, 1), density=False)
    frac = hist / max(hist.sum(), 1)
    # total variation distance from uniform: 0 is perfect
    tv = float(0.5 * np.abs(frac - 1.0 / n_bins).sum())
    return {"pit_fractions": frac.tolist(), "bin_edges": edges.tolist(),
            "tv_from_uniform": tv}
