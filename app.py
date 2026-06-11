from fastapi import FastAPI, File, UploadFile, Form, Body
from pydantic import BaseModel
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import torch
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
import rembg
import os
import time
from io import BytesIO
import logging
from typing import List



class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

app = FastAPI()

NUM_RENDER_VIEWS = 24

async def load_models():
    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    timer.start("Initializing model")
    # 加载模型
    model = TSR.from_pretrained(
        "/data/.cache/huggingface/hub/models--stabilityai--TripoSR/snapshots/9700b06c1641864ecbbe5eb0d89b967f3045cd5e",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)
    timer.end("Initializing model")
    return model, device

@app.on_event("startup")
async def startup_event():
    # 在应用启动时加载模型，并存储到app.state中
    app.state.model, app.state.device = await load_models()


def check_existence(image_index: int, output_dir: str, save_format: str, render: bool):
    """检查模型和渲染文件是否存在，如果都存在返回True，否则返回False"""
    mesh_path = os.path.join(output_dir, str(image_index), f"mesh.{save_format}")
    render_path = os.path.join(output_dir, str(image_index), "render_000.png")  # 仅检查第一帧作为示例
    if os.path.exists(mesh_path) and (not render or os.path.exists(render_path)):
        return True
    return False


class ModelRequest(BaseModel):
    image_paths: List[str] = ["examples/chair.png"]
    remove_bg: bool = True
    foreground_ratio: float = 0.85
    render: bool = True
    save_format: str = "glb"
    output_dir: str = "/data/TripoSR/output"
    is_skip_exist: bool = True


@app.post("/generate-3d-model/")
async def generate_3d_model(request: ModelRequest):
    output_dir = request.output_dir
    os.makedirs(output_dir, exist_ok=True)
    mesh_paths = []
    render_path_list:List[List[str]] = [[] for _ in range(len(request.image_paths))]

    timer.start("Processing images")
    images = []
    if not request.remove_bg:
        rembg_session = None
    else:
        rembg_session = rembg.new_session()


    for i, image_path in enumerate(request.image_paths):
        # 检查是否跳过已存在的模型和渲染
        if request.is_skip_exist and check_existence(i, request.output_dir, request.save_format, request.render):
            logging.info(f"Skipping existing model and render for image {i + 1}")
            mesh_path = os.path.join(request.output_dir, str(i), f"mesh.{request.save_format}")
            mesh_paths.append(mesh_path)
            if request.render:
                # 假设渲染帧和视频已经存在，则按照NUM_RENDER_VIEWS添加Multiview路径
                for ri in range(NUM_RENDER_VIEWS):
                    render_path = os.path.join(output_dir, str(i), f"render_{ri:03d}.png")
                    # render_image.save(render_path)
                    render_path_list[i].append(render_path)
            continue

        if not request.remove_bg:
            image = np.array(Image.open(image_path).convert("RGB"))
        else:
            image = remove_background(Image.open(image_path), rembg_session)
            image = resize_foreground(image, request.foreground_ratio)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            if not os.path.exists(os.path.join(output_dir, str(i))):
                os.makedirs(os.path.join(output_dir, str(i)))
            image.save(os.path.join(output_dir, str(i), f"input.png"))
        images.append(image)
        timer.end("Processing images")

    # for i, image in enumerate(images):
        logging.info(f"Running image {i + 1}/{len(images)} ...")

        # 处理图像
        timer.start("Running model")
        with torch.no_grad():
            scene_codes = app.state.model([image], device=app.state.device)
        timer.end("Running model")

        # 渲染（如果指定）
        if request.render:
            timer.start("Rendering")
            render_images = app.state.model.render(scene_codes, n_views=NUM_RENDER_VIEWS, return_type="pil")
            for ri, render_image in enumerate(render_images[0]):
                render_path = os.path.join(output_dir, str(i), f"render_{ri:03d}.png")
                render_image.save(render_path)
                render_path_list[i].append(render_path)
                
            save_video(render_images[0], os.path.join(output_dir,str(i), "render.mp4"), fps=NUM_RENDER_VIEWS)
            timer.end("Rendering")

        # 导出3D模型
        timer.start("Exporting mesh")
        meshes = app.state.model.extract_mesh(scene_codes)
        mesh_path = os.path.join(output_dir, str(i), f"mesh.{request.save_format}")
        meshes[0].export(mesh_path)
        mesh_paths.append(mesh_path)
        timer.end("Exporting mesh")
    
    result_json={
        "mesh_paths": mesh_paths,
        "render_path_list": render_path_list
    }
    return result_json
