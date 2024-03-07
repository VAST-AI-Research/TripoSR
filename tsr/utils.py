import importlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import PIL.Image
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from omegaconf import DictConfig, OmegaConf
from PIL import Image


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg


def find_class(cls_string):
    module_string = ".".join(cls_string.split(".")[:-1])
    cls_name = cls_string.split(".")[-1]
    module = importlib.import_module(module_string, package=None)
    cls = getattr(module, cls_name)
    return cls


def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)


class BaseModule(nn.Module):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        raise NotImplementedError


class ImagePreprocessor:
    def convert_and_resize(
        self,
        image: Union[PIL.Image.Image, np.ndarray, torch.Tensor],
        size: int,
    ):
        if isinstance(image, PIL.Image.Image):
            image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
            else:
                image = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            pass

        batched = image.ndim == 4

        if not batched:
            image = image[None, ...]
        image = F.interpolate(
            image.permute(0, 3, 1, 2),
            (size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).permute(0, 2, 3, 1)
        if not batched:
            image = image[0]
        return image

    def __call__(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        size: int,
    ) -> Any:
        if isinstance(image, (np.ndarray, torch.FloatTensor)) and image.ndim == 4:
            image = self.convert_and_resize(image, size)
        else:
            if not isinstance(image, list):
                image = [image]
            image = [self.convert_and_resize(im, size) for im in image]
            image = torch.stack(image, dim=0)
        return image


def rays_intersect_bbox(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    radius: float,
    near: float = 0.0,
    valid_thresh: float = 0.01,
):
    input_shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.view(-1, 3), rays_d.view(-1, 3)
    rays_d_valid = torch.where(
        rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d
    )
    if type(radius) in [int, float]:
        radius = torch.FloatTensor(
            [[-radius, radius], [-radius, radius], [-radius, radius]]
        ).to(rays_o.device)
    radius = (
        1.0 - 1.0e-3
    ) * radius  # tighten the radius to make sure the intersection point lies in the bounding box
    interx0 = (radius[..., 1] - rays_o) / rays_d_valid
    interx1 = (radius[..., 0] - rays_o) / rays_d_valid
    t_near = torch.minimum(interx0, interx1).amax(dim=-1).clamp_min(near)
    t_far = torch.maximum(interx0, interx1).amin(dim=-1)

    # check wheter a ray intersects the bbox or not
    rays_valid = t_far - t_near > valid_thresh

    t_near[torch.where(~rays_valid)] = 0.0
    t_far[torch.where(~rays_valid)] = 0.0

    t_near = t_near.view(*input_shape, 1)
    t_far = t_far.view(*input_shape, 1)
    rays_valid = rays_valid.view(*input_shape)

    return t_near, t_far, rays_valid


def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    if chunk_size <= 0:
        return func(*args, **kwargs)
    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert (
        B is not None
    ), "No tensor found in args or kwargs, cannot determine batch size."
    out = defaultdict(list)
    out_type = None
    # max(1, B) to support B == 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[
                arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ],
            **{
                k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for k, arg in kwargs.items()
            },
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(
                f"Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}."
            )
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all([vv is None for vv in v]):
            # allow None in return value
            out_merged[k] = None
        elif all([isinstance(vv, torch.Tensor) for vv in v]):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(
                f"Unsupported types in return value of func: {[type(vv) for vv in v if not isinstance(vv, torch.Tensor)]}"
            )

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in [tuple, list]:
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


ValidScale = Union[Tuple[float, float], torch.FloatTensor]


def scale_tensor(dat: torch.FloatTensor, inp_scale: ValidScale, tgt_scale: ValidScale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, torch.FloatTensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "softplus":
        return lambda x: F.softplus(x)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
    normalize: bool = True,
) -> torch.FloatTensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)

    if normalize:
        directions = F.normalize(directions, dim=-1)

    return directions


def get_rays(
    directions,
    c2w,
    keepdim=False,
    normalize=False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_spherical_cameras(
    n_views: int,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float,
    height: int,
    width: int,
):
    azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
    elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)

    fovy = torch.full_like(elevation_deg, fovy_deg) * math.pi / 180

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    # get directions by dividing directions_unit_focal by focal length
    focal_length = 0.5 * height / torch.tan(0.5 * fovy)
    directions_unit_focal = get_ray_directions(
        H=height,
        W=width,
        focal=1.0,
    )
    directions = directions_unit_focal[None, :, :, :].repeat(n_views, 1, 1, 1)
    directions[:, :, :, :2] = (
        directions[:, :, :, :2] / focal_length[:, None, None, None]
    )
    # must use normalize=True to normalize directions here
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)

    return rays_o, rays_d


def remove_background(
    image: PIL.Image.Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> PIL.Image.Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def resize_foreground(
    image: PIL.Image.Image,
    ratio: float,
) -> PIL.Image.Image:
    image = np.array(image)
    assert image.shape[-1] == 4
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    new_image = PIL.Image.fromarray(new_image)
    return new_image


def save_video(
    frames: List[PIL.Image.Image],
    output_path: str,
    fps: int = 30,
):
    # use imageio to save video
    frames = [np.array(frame) for frame in frames]
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def to_gradio_3d_orientation(mesh):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    return mesh
