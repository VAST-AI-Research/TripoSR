from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage import measure

class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Callable = marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx, _, __ = measure.marching_cubes(
            (level.detach().cpu() if level.is_cuda else level.detach()).numpy(),
            0.0)  # self.mc_func(level.detach(), 0.0)
        v_pos = torch.from_numpy(v_pos.copy()).type(torch.FloatTensor).to(level.device)
        t_pos_idx = torch.from_numpy(t_pos_idx.copy()).type(torch.LongTensor).to(level.device)
        v_pos = v_pos[..., [0, 1, 2]]
        t_pos_idx = t_pos_idx[..., [1, 0, 2]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos, t_pos_idx
