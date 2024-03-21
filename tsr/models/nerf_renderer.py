from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torchmcubes import marching_cubes

from ..utils import (
    BaseModule,
    chunk_batch,
    get_activation,
    rays_intersect_bbox,
    scale_tensor,
)


class TriplaneNeRFRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float

        feature_reduction: str = "concat"
        density_activation: str = "trunc_exp"
        density_bias: float = -1.0
        color_activation: str = "sigmoid"
        num_samples_per_ray: int = 128
        randomized: bool = False

    cfg: Config

    def configure(self) -> None:
        assert self.cfg.feature_reduction in ["concat", "mean"]
        self.chunk_size = 0

    def set_chunk_size(self, chunk_size: int):
        assert (
            chunk_size >= 0
        ), "chunk_size must be a non-negative integer (0 for no chunking)."
        self.chunk_size = chunk_size

    def interpolate_triplane(self, triplane: torch.Tensor, resolution: int):
        coords = torch.linspace(-1.0, 1.0, resolution, device = triplane.device)
        x, y = torch.meshgrid(coords, coords, indexing="ij")
        verts2D = torch.cat([x.view(resolution, resolution,1), y.view(resolution, resolution,1)], dim = -1)
        verts2D = verts2D.expand(3, -1, -1, -1)
        return F.grid_sample(triplane, verts2D, align_corners=False,mode="bilinear") # [3 40 H W] xy xz yz

    def block_based_marchingcube(self, decoder: torch.nn.Module, triplane: torch.Tensor, resolution: int, threshold, block_resolution = 128) -> torch.Tensor:
        resolution += 1 # sample 1 more line of density, 1024 + 1 == 1025, 0 mapping to -1.0f, 512 mapping to 0.0f, 1025 mapping to 1.0f,  for better floating point precision.
        block_size = 2.0 * block_resolution / (resolution - 1)
        voxel_size = block_size / block_resolution
        interpolated = self.interpolate_triplane(triplane, resolution)

        pos_list = []
        indices_list = []
        for x in range(0, resolution - 1, block_resolution):
            size_x = resolution - x if x + block_resolution >= resolution else block_resolution + 1 # sample 1 more line of density, so marching cubes resolution match block_resolution
            for y in range(0, resolution - 1, block_resolution):
                size_y = resolution - y if y + block_resolution >= resolution else block_resolution + 1
                for z in range(0, resolution - 1, block_resolution):
                    size_z = resolution - z if z + block_resolution >= resolution else block_resolution + 1
                    xyplane = interpolated[0:1, :, x:x+size_x, y:y+size_y].expand(size_z, -1, -1, -1, -1).permute(3, 4, 0, 1, 2)
                    xzplane = interpolated[1:2, :, x:x+size_x, z:z+size_z].expand(size_y, -1, -1, -1, -1).permute(3, 0, 4, 1, 2)
                    yzplane = interpolated[2:3, :, y:y+size_y, z:z+size_z].expand(size_x, -1, -1, -1, -1).permute(0, 3, 4, 1, 2)
                    sz = size_x * size_y * size_z
                    out = torch.cat([xyplane, xzplane, yzplane], dim=3).view(sz, 3, -1)

                    if self.cfg.feature_reduction == "concat":
                        out = out.view(sz, -1)
                    elif self.cfg.feature_reduction == "mean":
                        out = reduce(out, "N Np Cp -> N Cp", Np=3, reduction="mean")
                    else:
                        raise NotImplementedError
                    net_out = decoder(out)
                    out = None # discard samples
                    density = net_out["density"]
                    net_out = None # discard colors
                    density = get_activation(self.cfg.density_activation)(density + self.cfg.density_bias).view(size_x, size_y, size_z)
                    try: # now do the marching cube
                        v_pos, indices = marching_cubes(density.detach(), threshold)
                    except AttributeError:
                        print("torchmcubes was not compiled with CUDA support, use CPU version instead.")
                        v_pos, indices = self.mc_func(density.detach().cpu(), 0.0)
                    offset = torch.tensor([x * voxel_size - 1.0, y * voxel_size - 1.0, z * voxel_size - 1.0], device = triplane.device)
                    v_pos = v_pos[..., [2, 1, 0]] * voxel_size + offset
                    
                    indices_list.append(indices)
                    pos_list.append(v_pos)
                    
        vertex_count = 0
        for i in range(0, len(pos_list)):
            indices_list[i] += vertex_count
            vertex_count += pos_list[i].size(0)
        
        return torch.cat(pos_list), torch.cat(indices_list)

    def query_triplane(
        self,
        decoder: torch.nn.Module,
        positions: torch.Tensor,
        triplane: torch.Tensor,
        scale_pos = True
    ) -> Dict[str, torch.Tensor]:
        input_shape = positions.shape[:-1]
        positions = positions.view(-1, 3)

        # positions in (-radius, radius)
        # normalized to (-1, 1) for grid sample
        if scale_pos:
            positions = scale_tensor(
                positions, (-self.cfg.radius, self.cfg.radius), (-1, 1)
            )

        def _query_chunk(x):
            indices2D: torch.Tensor = torch.stack(
                (x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]),
                dim=-3,
            )
            out: torch.Tensor = F.grid_sample(
                rearrange(triplane, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
                rearrange(indices2D, "Np N Nd -> Np () N Nd", Np=3),
                align_corners=False,
                mode="bilinear",
            )
            if self.cfg.feature_reduction == "concat":
                out = rearrange(out, "Np Cp () N -> N (Np Cp)", Np=3)
            elif self.cfg.feature_reduction == "mean":
                out = reduce(out, "Np Cp () N -> N Cp", Np=3, reduction="mean")
            else:
                raise NotImplementedError

            net_out: Dict[str, torch.Tensor] = decoder(out)
            return net_out

        if self.chunk_size > 0:
            net_out = chunk_batch(_query_chunk, self.chunk_size, positions)
        else:
            net_out = _query_chunk(positions)

        net_out["density_act"] = get_activation(self.cfg.density_activation)(
            net_out["density"] + self.cfg.density_bias
        )
        net_out["color"] = get_activation(self.cfg.color_activation)(
            net_out["features"]
        )

        net_out = {k: v.view(*input_shape, -1) for k, v in net_out.items()}

        return net_out

    def _forward(
        self,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        **kwargs,
    ):
        rays_shape = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)
        n_rays = rays_o.shape[0]

        t_near, t_far, rays_valid = rays_intersect_bbox(rays_o, rays_d, self.cfg.radius)
        t_near, t_far = t_near[rays_valid], t_far[rays_valid]

        t_vals = torch.linspace(
            0, 1, self.cfg.num_samples_per_ray + 1, device=triplane.device
        )
        t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
        z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]  # (N_rays, N_samples)

        xyz = (
            rays_o[:, None, :] + z_vals[..., None] * rays_d[..., None, :]
        )  # (N_rays, N_sample, 3)

        mlp_out = self.query_triplane(
            decoder=decoder,
            positions=xyz,
            triplane=triplane,
        )

        eps = 1e-10
        # deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
        deltas = t_vals[1:] - t_vals[:-1]  # (N_rays, N_samples)
        alpha = 1 - torch.exp(
            -deltas * mlp_out["density_act"][..., 0]
        )  # (N_rays, N_samples)
        accum_prod = torch.cat(
            [
                torch.ones_like(alpha[:, :1]),
                torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1),
            ],
            dim=-1,
        )
        weights = alpha * accum_prod  # (N_rays, N_samples)
        comp_rgb_ = (weights[..., None] * mlp_out["color"]).sum(dim=-2)  # (N_rays, 3)
        opacity_ = weights.sum(dim=-1)  # (N_rays)

        comp_rgb = torch.zeros(
            n_rays, 3, dtype=comp_rgb_.dtype, device=comp_rgb_.device
        )
        opacity = torch.zeros(n_rays, dtype=opacity_.dtype, device=opacity_.device)
        comp_rgb[rays_valid] = comp_rgb_
        opacity[rays_valid] = opacity_

        comp_rgb += 1 - opacity[..., None]
        comp_rgb = comp_rgb.view(*rays_shape, 3)

        return comp_rgb

    def forward(
        self,
        decoder: torch.nn.Module,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if triplane.ndim == 4:
            comp_rgb = self._forward(decoder, triplane, rays_o, rays_d)
        else:
            comp_rgb = torch.stack(
                [
                    self._forward(decoder, triplane[i], rays_o[i], rays_d[i])
                    for i in range(triplane.shape[0])
                ],
                dim=0,
            )

        return comp_rgb

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
