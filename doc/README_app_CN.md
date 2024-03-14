# FastAPI TRIPOSR 3D Model Generation

本仓库开发了一个FastAPI应用(`app.py`)，专为[TripoSR](https://github.com/VAST-AI-Research/TripoSR.git)项目作为后端应用程序。TripoSR是截至2024年3月4日为止速度最快、质量最高的图像到3D模型生成器。引入这个FastAPI应用旨在通过提供一系列图像处理和3D模型生成功能，助力开发者轻松构建Image-to-3D应用。

## 优势

- **高效性**：避免每次请求时重新加载模型权重，大大减少处理时间和资源消耗。
- **存在性检查**：自动检测输出目录是否已包含生成的数据，并在存在时跳过处理，优化工作流程并提高存储利用率。

## [TripoSR](https://github.com/VAST-AI-Research/TripoSR.git)的关键特性
- **快的离谱!!!**: 实验室环境A6000GPU下测速: 1.7s推理 2s导出3D文件 17s渲染(所以不用看效果可以关掉render)
- **背景去除**：自动从输入图像中移除背景，专注于主体对象进行3D模型生成。
- **图像缩放**：调整前景比例以确保在最佳条件下生成3D模型。
- **3D模型渲染**：提供可选的渲染支持能力来可视化生成的3D模型，提升开发和测试过程体验。
- **灵活的输出格式**：支持多种3D模型输出格式，包括`.glb`和`.obj`，满足不同应用需求的多样性。
- **性能监控**：整合自定义计时器工具跟踪和优化3D模型生成过程的性能。

## 安装与使用

要将FastAPI TRIPOSR 3D Model Generation应用集成到TripoSR项目，请按照以下步骤操作，确保系统上已安装Python 3.8+版本。

### 第1步：部署TripoSR

首先，从GitHub克隆TripoSR仓库并进入项目目录：

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
```

### 第2步：集成FastAPI应用

将此仓库中的`app.py`文件放置到TripoSR项目目录中。您可以直接从此仓库下载`app.py`或如果已在本地机器上复制它。

### 第3步：安装依赖项

在TripoSR项目目录下，确保已按照其要求安装所有TripoSR依赖项。然后，安装FastAPI和其他由`app.py`需要的附加依赖项：

```bash
pip install fastapi uvicorn python-multipart
```

### 第4步：启动FastAPI服务器

现在`app.py`已经成为TripoSR目录的一部分并且所有依赖项已经安装，您可以使用Uvicorn启动FastAPI服务器。在终端中，在TripoSR项目目录下运行以下命令：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

该命令将在端口8000上启动FastAPI应用（如果您需要可以使用其他端口），并启用开发环境下的实时重载。

### 发送请求

要从图像生成3D模型，请向`/generate-3d-model/`端点发送一个POST请求，附带一个JSON负载指定图像路径和其他参数。这里是一个使用`curl`的例子：

```bash
curl -X 'POST' \
  'http://localhost:8000/generate-3d-model/' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_paths": ["/data/TripoSR/examples/flamingo.png","/data/TripoSR/examples/hamburger.png","/data/TripoSR/examples/teapot.png"],
  "remove_bg": true,
  "foreground_ratio": 0.85,
  "render": true,
  "save_format": "glb",
  "output_dir": "/data/TripoSR/output",
  "is_skip_exist": true
}'
```

### 请求体格式说明

请求体中使用的数据结构遵循`ModelRequest`类的定义：

```python
class ModelRequest(BaseModel):
    image_paths: List[str] = ["examples/chair.png"]  # 图像文件路径列表
    remove_bg: bool = True                          # 是否移除背景
    foreground_ratio: float = 0.85                 # 前景比例
    render: bool = True                            # 是否渲染3D模型
    save_format: str = "glb"                       # 输出模型的文件格式
    output_dir: str = "/data/TripoSR/output"       # 输出目录
    is_skip_exist: bool = True                     # 如果目标文件已存在则跳过处理
```

### 响应

响应包含生成的3D模型及任何渲染图像的路径：

```json
{
  "mesh_paths": ["/data/TripoSR/output/0/mesh.glb", "/data/TripoSR/output/1/mesh.glb", "/data/TripoSR/output/2/mesh.glb"],
  "render_path_list": [["/data/TripoSR/output/0/render_000.png"], [...], [...]]
}
```

