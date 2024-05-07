# TripoSR <a href="https://huggingface.co/stabilityai/TripoSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a> <a href="https://huggingface.co/spaces/stabilityai/TripoSR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> <a href="https://huggingface.co/papers/2403.02151"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-Huggingface-orange"></a> <a href="https://arxiv.org/abs/2403.02151"><img src="https://img.shields.io/badge/Arxiv-2403.02151-B31B1B.svg"></a> <a href="https://discord.gg/mvS9mCfMnQ"><img src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>

<div align="center">
  <img src="figures/teaser800.gif" alt="Teaser Video">
</div>

This is the official codebase for **TripoSR**, a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).
<br><br>
Leveraging the principles of the [Large Reconstruction Model (LRM)](https://yiconghong.me/LRM/), TripoSR brings to the table key advancements that significantly boost both the speed and quality of 3D reconstruction. Our model is distinguished by its ability to rapidly process inputs, generating high-quality 3D models in less than 0.5 seconds on an NVIDIA A100 GPU. TripoSR has exhibited superior performance in both qualitative and quantitative evaluations, outperforming other open-source alternatives across multiple public datasets. The figures below illustrate visual comparisons and metrics showcasing TripoSR's performance relative to other leading models. Details about the model architecture, training process, and comparisons can be found in this [technical report](https://arxiv.org/abs/2403.02151).

<!--
<div align="center">
  <img src="figures/comparison800.gif" alt="Teaser Video">
</div>
-->
<p align="center">
    <img width="800" src="figures/visual_comparisons.jpg"/>
</p>

<p align="center">
    <img width="450" src="figures/scatter-comparison.png"/>
</p>


The model is released under the MIT license, which includes the source code, pretrained models, and an interactive online demo. Our goal is to empower researchers, developers, and creatives to push the boundaries of what's possible in 3D generative AI and 3D content creation.

## Getting Started
### Installation
- Python >= 3.8
- Install CUDA if available
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) **[Please make sure that the locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.]**
- Update setuptools by `pip install --upgrade setuptools`
- Install other dependencies by `pip install -r requirements.txt`

### Manual Inference
```sh
python run.py examples/chair.png --output-dir output/
```
This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.

If you would like to output a texture instead of vertex colors, use the `--bake-texture` option. You may also use `--texture-resolution` to specify the resolution in pixels of the output texture.

For detailed usage of this script, use `python run.py --help`.

### Local Gradio App
```sh
python gradio_app.py
```

## Troubleshooting
> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that 

- The locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.
- `setuptools>=49.6.0`. If not, upgrade by `pip install --upgrade setuptools`.

Then re-install `torchmcubes` by:

```sh
pip uninstall torchmcubes
pip install git+https://github.com/tatsy/torchmcubes.git
```

## Citation
```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```
