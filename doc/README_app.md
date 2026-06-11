# FastAPI TRIPOSR 3D Model Generation
[中文版](./README_CN.md)

This repository introduces a FastAPI application (`app.py`) specifically designed as a backend application for the [TripoSR](https://github.com/VAST-AI-Research/TripoSR.git) project. As of March 4, 2024, TripoSR stands out as the fastest and highest-quality Image-to-3D model generator. The introduction of this FastAPI application aims to facilitate developers in creating Image-to-3D applications by offering a suite of image processing and 3D model generation functionalities.

## Advantages

- **Efficiency**: Avoids reloading model weights for every request, significantly reducing processing time and resource consumption.
- **Existence Check**: Automatically detects if the output directory already contains generated data and skips processing when present, optimizing workflow and improving storage utilization.

## Key Features of [TripoSR](https://github.com/VAST-AI-Research/TripoSR.git)

- **Blazing Fast!!!**: In lab conditions with an A6000 GPU, benchmarking shows: 1.7s for inference, 2s for exporting 3D files, and 17s for rendering (rendering can be turned off if visual results are not needed).
- **Background Removal**: Automatically removes backgrounds from input images, focusing on the main subject for 3D model generation.
- **Image Resizing**: Adjusts the foreground ratio to ensure optimal conditions for generating 3D models.
- **3D Model Rendering**: Offers optional rendering support to visualize the generated 3D models, enhancing the development and testing experience.
- **Flexible Output Formats**: Supports multiple output formats for 3D models, including `.glb` and `.obj`, catering to diverse application requirements.
- **Performance Monitoring**: Incorporates a custom timer utility to track and optimize the performance of the 3D model generation process.

## Installation & Usage

To integrate the FastAPI TRIPOSR 3D Model Generation application into the TripoSR project, follow these steps with Python 3.8+ installed on your system.

### Step 1: Deploy TripoSR

Firstly, clone the TripoSR repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TriposR
```

### Step 2: Integrate the FastAPI Application

Place the `app.py` file from this repository into the TripoSR project directory. You can download it directly from this repository or copy it if you already have it locally.

### Step 3: Install Dependencies

Within the TripoSR project directory, ensure that all dependencies required by TripoSR are installed as per its specifications. Then, install FastAPI and any additional dependencies needed by `app.py`:

```bash
pip install fastapi uvicorn python-multipart
```

### Step 4: Launch the FastAPI Server

Now that `app.py` is part of the TripoSR directory and all dependencies are installed, you can start the FastAPI server using Uvicorn. Run the following command in the terminal within the TripoSR project directory:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

This command starts the FastAPI application on port 8000 (you can use a different port if desired) and enables live reloading for development purposes.

### Sending a Request

To generate 3D models from images, send a POST request to the `/generate-3d-model/` endpoint with a JSON payload specifying the image paths and other parameters. Here's an example using `curl`:

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

### Body Format Specification

The structure used in the request body adheres to the definition of the `ModelRequest` class:

```python
class ModelRequest(BaseModel):
    image_paths: List[str] = ["examples/chair.png"]  # List of image file paths
    remove_bg: bool = True                          # Whether to remove the background
    foreground_ratio: float = 0.85                 # Foreground ratio adjustment
    render: bool = True                            # Whether to render the 3D model
    save_format: str = "glb"                       # Output file format for the model
    output_dir: str = "/data/TripoSR/output"       # Output directory
    is_skip_exist: bool = True                     # Skip processing if target file exists
```

### Response

The response includes the paths to the generated 3D models and any rendered images:

```json
{
  "mesh_paths": ["/data/TripoSR/output/0/mesh.glb", "/data/TripoSR/output/1/mesh.glb", "/data/TripoSR/output/2/mesh.glb"],
  "render_path_list": [["/data/TripoSR/output/0/render_000.png"], [...], [...]]
}
```
