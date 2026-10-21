"""The official 4DVarNet on the gridded forecast scenes: one variational solve per month, read at the queries.

``fourdvar_synthetic.OfficialFourDVarNet`` (the authors' ``GradSolver``, ``ConvLstmGradModel`` and
``BilinAEPriorCost``, executed unchanged from the pinned 4dvarnet-starter commit) behind the
``encode`` / ``decode`` interface of ``grid_kriging_ocean.GridKrigingModel``, so a runner and the
sealed scorers (``sealed_scoring.collect``) treat it like any gridded model:

  encode   the month's input profiles (the scene's ``obs_*`` tokens: exact positions and per-level
           T/S values) and the dense SST/SSS/SSH grids of the input month form the 4DVarNet batch.
           The state is the 1-degree grid of T at every level, S at every level and the three
           surface fields; the observation cost reads it bilinearly at every profile position and
           compares the surface channels with the grids where they exist; the solver starts from the
           profiles' cell means and the surface grids, zero elsewhere (the official initialisation).
  decode   a query reads its level's T and S from the state at its position (bilinear, periodic in
           longitude). At a forecast lead the input month's state is read at the target month's
           queries: a persisted 4DVarNet reconstruction, like the persisted OI.

Training follows the official ``Lit4dVarNet.step``: 50 x the reconstruction MSE (here the held-out
query cells, channel-balanced like our models') + 1 x the prior cost of the output state
(``prior_cost`` of ``encode``); its Sobel term needs dense truth and is dropped. In evaluation the
official solver ends with the prior autoencoder's projection; ``output`` picks that ("projection")
or the state after the solver steps, the training-mode output ("solver"). The solver is
deterministic: ``std`` is NaN and only the regression metrics apply.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .fourdvar_synthetic import FourDVarConfig, OfficialFourDVarNet, make_batch

OUTPUTS = ("solver", "projection")


class FourDVarGridModel(nn.Module):

    def __init__(self, repository, config: FourDVarConfig, output: str = "solver"):
        super().__init__()
        if output not in OUTPUTS:
            raise ValueError(f"output is one of {OUTPUTS}")
        if not config.surface:
            raise ValueError("the scenes always carry the surface grids; configure surface=True")
        self.net = OfficialFourDVarNet(repository, config)
        self.config, self.output = config, output

    @property
    def official_source(self):
        return self.net.official_source

    def batch(self, scene: Mapping[str, Tensor]):
        """The official batch of a scene's input month (profiles NaN where not observed)."""
        c = self.config
        coord = scene["obs_coord"]
        n = coord.shape[0] // c.levels
        coord = coord.reshape(n, c.levels, 4)
        values = scene["obs_innovation"].reshape(n, c.levels, 2).float()
        valid = scene["obs_valid"].reshape(n, c.levels, 2).bool() & torch.isfinite(values)
        observed = torch.where(valid, values, torch.full_like(values, float("nan")))
        surface = scene["surface_grid"].float()
        if surface.shape != (3, c.height, c.width):
            raise ValueError("surface_grid must be [3, height, width]")
        return make_batch(observed, coord[:, 0, 0].float(), coord[:, 0, 1].float(), c, surface[None])

    def solve(self, batch):
        """``GradSolver.forward`` in evaluation, step for step, keeping the state before its final
        projection: (state after the solver steps, the official output)."""
        s = self.net.solver
        with torch.set_grad_enabled(True):
            state = s.init_state(batch)
            s.grad_mod.reset_state(batch.input)
            for step in range(s.n_step):
                state = s.solver_step(state, batch, step=step)
                state = state.detach().requires_grad_(True)
            projected = s.prior_cost.forward_ae(state)
        return state.detach(), projected.detach()

    def encode(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        batch = self.batch(scene)
        lead = int(scene["lead"]) if "lead" in scene else 0
        if self.training:
            s = self.net.solver
            state = self.net(batch)
            return {"state": state, "prior_cost": s.prior_cost(s.init_state(batch, state)), "lead": lead}
        solver, projection = self.solve(batch)
        return {"state": {"solver": solver, "projection": projection}[self.output], "solver": solver,
                "projection": projection, "prior_cost": solver.new_zeros(()), "lead": lead}

    def decode(self, scene: Mapping[str, Tensor], state: Mapping[str, Any], output: str | None = None) -> dict[str, Any]:
        field = state["state"] if output is None else state[output]
        coord = scene["query_coord"]
        mean = self.net.query(field, coord[:, 0].float(), coord[:, 1].float(), scene["query_level"])[0]
        return {"mean": mean, "std": torch.full_like(mean, float("nan")), "aux_loss": state["prior_cost"]}

    def forward(self, scene: Mapping[str, Tensor]) -> dict[str, Any]:
        return self.decode(scene, self.encode(scene))
