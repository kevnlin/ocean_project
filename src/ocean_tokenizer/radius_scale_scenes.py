"""Two options on the matched synthetic scenes: a neighbour radius, a local scale.

Both answer review points on the OI-anchored model. With neither set, a scene
is exactly the parent's.

``radius_km``  The local branch reads the 32 nearest input profiles of a query
               however far away they are. With a radius, a neighbour beyond it
               is marked invalid; the model already masks invalid neighbours,
               and a query left with none keeps OI plus the global residual.
               The frozen OI anchor and the shared latent are untouched.

``scale``      ``(P, L, 2)`` positive factors, one per profile, level and
               variable (``local_scale.lookup`` of a table). The local branch
               then works in units of the query's own scale: the innovations of
               all its neighbours, its anchor and its training target are
               divided by the scale at the query. The anchor is still the exact
               frozen OI, computed in per-depth z units from the unscaled
               values; the shared latent keeps reading per-depth z values.
               ``to_z`` converts a prediction back, so scores are in the
               parent's units.

``scale_by``   ``"query"`` is the above. ``"profile"`` is the first design,
               kept so its runs can be replayed: every profile's innovations
               (neighbours and latent tokens) divided by that profile's own
               scale. With a table that is constant inside 5-degree cells this
               puts two neighbours on either side of a cell edge in different
               units and distorts the field the local branch interpolates;
               plain OI on such values loses about half its skill.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .audit_tools import cohort_path
from .point_baselines import CH, Scores
from .synthetic_latent_experiment import SyntheticOceanScenes


class RadiusScaleScenes(SyntheticOceanScenes):

    def __init__(self, *args, radius_km=None, scale=None, scale_by="query", **kwargs):
        super().__init__(*args, **kwargs)
        self.climatology_reference = None
        if kwargs.get("cohort") is not None:
            self.use_cohort_reference(kwargs["cohort"], kwargs.get("oi_summary"))
        self.configure(radius_km, scale, scale_by)

    def use_cohort_reference(self, cohort, summary_path):
        """Another synthetic cohort: its own fixed baselines are the identity reference.

        The parent proves its evaluation cells by reproducing the climatology
        error stored with a model trained on the CESM2 cohort, which says
        nothing about another cohort. There the reference is the climatology
        row of that cohort's ``44_synth_argo_oi.py --cohort`` summary, which
        must have been computed on this very file.
        """
        if summary_path is None:
            raise ValueError("a named cohort needs its own fixed baselines (--oi-summary)")
        summary = json.loads(Path(summary_path).read_text())
        path = Path(cohort_path(str(self.root), "synthetic", cohort=cohort))
        if Path(summary["cohort"]["path"]).name != path.name:
            raise ValueError("these baselines were computed on another cohort")
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary["cohort"]["sha256"]:
            raise ValueError("the cohort file was rebuilt after its baselines were computed")
        self.climatology_reference = summary["baselines"]["climatology"]["scores"]
        self.metadata["cohort"] = cohort

    def identity_check(self, split, evs):
        reference = getattr(self, "climatology_reference", None)
        if reference is None:
            return super().identity_check(split, evs)
        scores = Scores(self.levels, self.norm.std)
        for ev in evs:
            for ch in CH:
                target = self.obs[ch][ev.target][ev.profile, ev.level]
                scores.add(ch, np.zeros(len(ev.level)), target, ev.level)
        zero, report = scores.result(), {}
        for ch in CH:
            report[ch] = {"n": zero[ch]["n"], "climatology_z": zero[ch]["climatology_z"],
                          "reference_n": reference[split][ch]["n"],
                          "reference_climatology_z": reference[split][ch]["rmse_z"]}
            if (report[ch]["n"] != report[ch]["reference_n"]
                    or abs(report[ch]["climatology_z"] / report[ch]["reference_climatology_z"] - 1.) > 1e-5):
                raise ValueError(f"identity check failed on {split} {ch}: {report[ch]}")
        return report

    def configure(self, radius_km=None, scale=None, scale_by="query"):
        if radius_km is not None and not radius_km > 0:
            raise ValueError("the local radius must be a positive number of km")
        if scale_by not in ("query", "profile"):
            raise ValueError("scale_by must be 'query' or 'profile'")
        self.radius_km = None if radius_km is None else float(radius_km)
        self.scale, digest = None, None
        self.scale_by = scale_by if scale is not None else None
        if scale is not None:
            scale = np.ascontiguousarray(scale, dtype=np.float32)
            if scale.shape != tuple(self.y.shape):
                raise ValueError(f"scale shape {scale.shape} differs from the values' "
                                 f"{tuple(self.y.shape)}")
            if not (np.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError("the scale must be positive and finite for every value")
            digest = hashlib.sha256(scale.tobytes()).hexdigest()
            self.scale = torch.as_tensor(scale, device=self.device)
        self.metadata.update(local_radius_km=self.radius_km, local_scale_sha256=digest,
                             local_scale_by=self.scale_by)

    def scene(self, source, query_rows, query_levels, analysis, context_profiles=256,
              context_seed=0, geometry=None, cached_baseline=None):
        if self.scale_by == "profile" and 0 < context_profiles < len(source):
            # the context rows are drawn inside the parent; with a full context
            # they are the source rows in order, which is what is rescaled below
            raise ValueError("a per-profile scale needs every source profile in the context")
        if geometry is None:
            geometry = self.geometry(source, query_rows, 32, query_levels)
        scene = super().scene(source, query_rows, query_levels, analysis, context_profiles,
                              context_seed, geometry, cached_baseline)
        if self.radius_km is not None:
            distance = scene["local_offsets"][..., :2].norm(dim=-1)
            scene["local_valid"] = scene["local_valid"] & (distance <= self.radius_km)[..., None]
        if self.scale is not None:
            q, lev = self.indices(query_rows), self.indices(query_levels)
            at_query = self.scale[q, lev]
            scene["baseline"] = scene["baseline"] / at_query
            if self.scale_by == "query":
                scene["local_innovation"] = scene["local_innovation"] / at_query[:, None]
            else:
                scene["local_innovation"] = (scene["local_innovation"]
                                             / self.scale[geometry["neighbors"], lev[:, None]])
                scene["obs_innovation"] = (scene["obs_innovation"]
                                           / self.scale[self.indices(source)].reshape(-1, 2))
        return scene

    def network_targets(self, rows, levels):
        """Targets in the units the network predicts; ``targets`` stays in z."""
        target, valid = self.targets(rows, levels)
        if self.scale is not None:
            target = target / self.scale[self.indices(rows), self.indices(levels)]
        return target, valid

    def to_z(self, value, rows, levels):
        """A predicted mean or standard deviation, network units -> per-depth z."""
        if self.scale is None:
            return value
        return value * self.scale[self.indices(rows), self.indices(levels)]


def fold_rows(rows, n_folds, seed, month):
    """A month's profiles in ``n_folds`` disjoint, near-equal, reproducible parts."""
    rows = np.asarray(rows)
    order = np.random.default_rng([int(seed), int(month)]).permutation(len(rows))
    return [np.sort(rows[order[fold::n_folds]]) for fold in range(n_folds)]


@torch.no_grad()
def crossfit_residual(data, months, n_folds=5, seed=20261008, chunk=4096):
    """``(P, L, 2)`` error of the frozen OI at every profile of ``months``.

    Each month's profiles are split into ``n_folds`` parts and each part is
    analysed from the others, so no profile is predicted from its own value.
    The synthetic cohort has one profile per float and 7,600 a month; five
    folds leave 6,080 inputs, the density of an evaluation month. Values are
    ``truth - analysis`` in per-depth z units, NaN where the truth is missing
    and for profiles of other months.
    """
    analysis = data.analysis()
    out = np.full(tuple(data.y.shape), np.nan, dtype=np.float32)
    month_index = np.asarray(data.c.month_index)
    for month in months:
        in_month = np.flatnonzero(month_index == int(month))
        for fold in fold_rows(in_month, n_folds, seed, month):
            source = np.setdiff1d(in_month, fold)
            # level-major, so a chunk touches few levels and the neighbour
            # search (one per level and variable) is not repeated needlessly
            rows = np.tile(fold, data.n_levels)
            levels = np.repeat(np.arange(data.n_levels), len(fold))
            for start in range(0, len(rows), chunk):
                r, lev = rows[start:start + chunk], levels[start:start + chunk]
                geometry = SyntheticOceanScenes.geometry(data, source, r, 32, lev)
                li = data.indices(lev)
                oi = analysis(geometry, li, data.innovation, data.xyz)
                out[r, lev] = (data.y[data.indices(r), li] - oi).cpu().numpy()
    return out
