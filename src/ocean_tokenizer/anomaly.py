"""Anomaly target + anomaly-space normalisation (Week-1 audit fix).

Motivation
----------
Training against the *absolute* z-scored field lets a model score well by simply
reproducing the (memorisable) climatological mean / seasonal cycle — most of the
field's variance *is* the climatology.  We instead regress the **anomaly**

    a(t) = field(t) - clim[month(t)]

against a **train-only monthly climatology** computed from the CESM2-LE ground
truth.  Two consequences:

* the model must learn the residual signal, not the memorisable mean;
* predicting *zero anomaly* recovers the climatology exactly, so the
  train-only climatology is a principled RMSE **floor** (the ~0.65 degC
  CESM2 self-climatology floor, distinct from the bias-dominated WOA prior).

Note the reconstruction RMSE is invariant to this reframing (the climatology
cancels in ``pred - true``); the change is in *what the network is trained to
predict* and in the *reference floor* we report against.

`AnomNorm` mirrors the ``Norm`` interface (``z3d`` / ``unz3d`` / ``zsurf``) with
an extra ``month`` argument, so the same feature-builders drive either an
absolute or an anomaly pipeline.
"""
from __future__ import annotations
import numpy as np

from . import config as C

VARS = ("TEMP", "SALT")


class Climatology:
    """Train-only monthly climatology of the CESM2-LE ground truth.

    Built from the *training* snapshots only (no test months leak in).  For a
    calendar month not present in the training split we fall back to the
    all-months mean so every test month is defined.
    """

    def __init__(self, train_fields: dict, surf_train: dict | None = None):
        months = np.asarray(train_fields["months"], dtype=int)   # (T,)
        self.clim = {}          # v -> (12, D, H, W)
        self.clim_surf = {}     # v -> (12, H, W)
        for v in VARS:
            x = train_fields[v]                                   # (T,D,H,W)
            glob = np.nanmean(x, axis=0)                          # (D,H,W)
            stack = np.empty((12,) + x.shape[1:], dtype="float32")
            for m in range(1, 13):
                sel = months == m
                if sel.any():
                    with np.errstate(invalid="ignore"):
                        stack[m - 1] = np.nanmean(x[sel], axis=0)
                else:
                    stack[m - 1] = glob
            self.clim[v] = stack
        surf_train = surf_train or {}
        for v in C.VARS_SURF:
            if v in surf_train and surf_train[v] is not None:
                s = surf_train[v]                                 # (T,H,W)
                glob = np.nanmean(s, axis=0)
                stack = np.empty((12,) + s.shape[1:], dtype="float32")
                for m in range(1, 13):
                    sel = months == m
                    stack[m - 1] = np.nanmean(s[sel], axis=0) if sel.any() else glob
                self.clim_surf[v] = stack

    def clim3d(self, v, month):     # month 1-12 -> (D,H,W)
        return self.clim[v][month - 1]

    def clim_surf3d(self, v, month):   # -> (H,W)
        return self.clim_surf[v][month - 1]


class AnomNorm:
    """Anomaly-space normaliser: z-scores ``field - clim[month]`` per depth.

    Statistics (per-depth anomaly mean/std) are estimated on the *training*
    anomalies only.  Interface parity with ``baselines.Norm`` — every method
    takes ``(v, arr, month)`` — so feature-builders are encoder-agnostic.
    """

    def __init__(self, clim: Climatology, train_fields: dict, surf_train: dict | None = None):
        self.clim = clim
        months = np.asarray(train_fields["months"], dtype=int)
        self.amean = {}
        self.astd = {}
        for v in VARS:
            x = train_fields[v]                                   # (T,D,H,W)
            a = x - clim.clim[v][months - 1]                      # (T,D,H,W) anomaly
            self.amean[v] = np.nan_to_num(
                np.nanmean(a, axis=(0, 2, 3))).astype("float32")  # (D,)
            sd = np.nanstd(a, axis=(0, 2, 3)).astype("float32")
            self.astd[v] = np.where(sd < 1e-6, 1.0, sd)
        self.asmean = {}
        self.asstd = {}
        surf_train = surf_train or {}
        for v in C.VARS_SURF:
            if v in surf_train and surf_train[v] is not None and v in clim.clim_surf:
                s = surf_train[v]
                a = s - clim.clim_surf[v][months - 1]
                self.asmean[v] = float(np.nan_to_num(np.nanmean(a)))
                sd = float(np.nanstd(a))
                self.asstd[v] = sd if sd > 1e-6 else 1.0

    # --- 3D volumetric (TEMP/SALT) ---
    def z3d(self, v, arr, month):
        a = arr - self.clim.clim3d(v, month)
        return (a - self.amean[v][:, None, None]) / self.astd[v][:, None, None]

    def unz3d(self, v, arr, month):
        a = arr * self.astd[v][:, None, None] + self.amean[v][:, None, None]
        return a + self.clim.clim3d(v, month)

    # --- surface (SST/SSS) ---
    def zsurf(self, v, arr, month):
        a = arr - self.clim.clim_surf3d(v, month)
        return (a - self.asmean[v]) / self.asstd[v]
