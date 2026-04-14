<h2 align="center" style="font-size:24px;">
  <b>G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior</b>
  <br>

  <b><i>ICLR 2026 </i></b>
</h2>

<p align="center">
    <a href="https://dali-jack.github.io/Junfeng-Ni/">Junfeng Ni </a><sup>1,2,*</sup>,
    <a href="https://yixchen.github.io/">Yixin Chen </a><sup>2,†,✉</sup>,
    <a href="https://github.com/isxiaohe/">Zhifei Yang </a><sup>3</sup>,
    <a href="https://yuliu-ly.github.io/">Yu Liu </a><sup>1,2</sup>,
    <a href="https://jason-aplp.github.io/Ruijie-Lu/">Ruijie Lu </a><sup>3</sup>,
    <a href="https://zhusongchun.net/">Song-Chun Zhu </a><sup>1,2,3</sup>,
    <a href="https://siyuanhuang.com/">Siyuan Huang </a><sup>2,✉</sup>
    <br>
    <sup>*</sup> Work done as an intern at BIGAI &nbsp
    <sup>†</sup> Project lead &nbsp
    <sup>✉</sup> Corresponding author
    <br>
    <sup>1</sup>Tsinghua University &nbsp
    <sup>2</sup>State Key Laboratory of General Artificial Intelligence, BIGAI &nbsp
    <sup>3</sup>Peking University
</p>

<p align="center">
    <a href='https://arxiv.org/abs/2510.12099'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://dali-jack.github.io/g4splat-web'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>

<p align="center">
    <img src="assets/teaser.png" width=90%>
</p>

<b>G4Splat</b> integrates <b>accurate geometry guidance with generative prior</b> to enhance 3D scene reconstruction, substantially improving both geometric fidelity and appearance quality in observed and unobserved regions.


## 1. Installation


### 1.1. Install dependencies

We provide a `pixi.toml` manifest for this repository. It is configured to match the
working `/home/rais/VerseCrafter` baseline as closely as practical on this machine:
Python 3.11 with PyTorch 2.10.0 / CUDA 12.8 wheels. Native build tasks reserve 4 CPU
cores by default and auto-detect a single `TORCH_CUDA_ARCH_LIST` target unless you
override `MAX_JOBS` or `TORCH_CUDA_ARCH_LIST` yourself.

```shell
# Optional: enable proxy only when your current network path needs it.
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897

pixi install
pixi run bootstrap
pixi run verify-env

# Detectron2 is only used for visualization utilities.
# Install it separately if you need those code paths.
pixi run install-detectron2
```

Notes:

- `bootstrap` includes `xformers`, `sam2`, `pytorch3d`, the 2DGS CUDA
  extensions, tetra-triangulation, and the MASt3R native extensions.
- If you want even stricter CPU throttling during native builds, override `MAX_JOBS`
  or use `PYTORCH3D_RESERVED_CPUS`, `LOCAL_EXT_RESERVED_CPUS`, `XFORMERS_RESERVED_CPUS`,
  and `DETECTRON2_RESERVED_CPUS`.
- When downloading checkpoints or datasets mirrored from Hugging Face, prefer checking
  ModelScope first. On this machine, ModelScope downloads are usually faster without
  proxy, so you can temporarily unset proxy variables for those commands.

If you prefer the original manual setup, follow the legacy steps below:

Please follow the instructions below to install the dependencies:

```shell
git clone https://github.com/DaLi-Jack/G4Splat.git --recursive

# Create and activate conda environment
conda create --name g4splat -y python=3.9
conda activate g4splat

# Install system dependencies via conda (required for compilation)
conda install cmake gmp cgal -c conda-forge

# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
pip install --no-build-isolation --no-deps https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip
# Detectron2 (used only for visualization)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then, install the 2D Gaussian splatting and adaptive tetrahedralization dependencies:

```shell
cd 2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../tetra-triangulation
cmake .
# you can specify your own cuda path
export CPATH=/usr/local/cuda-11.8/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.8/bin:$PATH
make 
pip install -e .
cd ../../../
```

Finally, install the MASt3R-SfM dependencies:

```shell
cd mast3r/asmk/cython
cythonize *.pyx
cd ..
pip install .
cd ../dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../
```


### 1.2. Download pretrained models

First, download the pretrained checkpoint for DepthAnythingV2. Several encoder sizes are available; We recommend using the `large` encoder:

```shell
mkdir -p ./Depth-Anything-V2/checkpoints/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P ./Depth-Anything-V2/checkpoints/
```

Then, download the MASt3R-SfM checkpoint:

```shell
mkdir -p ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./mast3r/checkpoints/
```

Then, download the MASt3R-SfM retrieval checkpoint:

```shell
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P ./mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P ./mast3r/checkpoints/
```

Then, download the SAM2 checkpoint:
```shell
mkdir -p ./checkpoint/sam2/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P ./checkpoint/sam2/
```

The plane-extraction stage defaults to the official SAM2.1 large config `configs/sam2.1/sam2.1_hiera_l.yaml` with checkpoint `./checkpoint/sam2/sam2.1_hiera_large.pt`.

For interleaved multi-view COLMAP image sets, `plane_excavator.py` now de-interleaves the sampled frames into per-view temporal sequences before SAM2 video tracking; the main training wrappers pass `--num_views 12`, while `select-gs-planes` inputs continue to auto-detect as a single sequence.

Finally, download the See3D checkpoint:
```shell
# Download the See3D checkpoint from HuggingFace first, then move it to the desired path
mv YOUR_LOCAL_PATH/MVD_weights ./checkpoint/MVD_weights
```


## 2.data
Please download the preprocessed [data](https://huggingface.co/datasets/JunfengNi/G4Splat) from HuggingFace and unzip in the `data` folder. The resulting folder structure should be:
```bash
└── G4Splat
  └── data
    ├── replica
        ├── scan ...
    ├── scannetpp
        ├── scan ...
    ├── deepblending
        ├── scan ...
    ├── denseview
        ├── scan1
```


## 3.Training and Evaluation
The evaluation code is integrated into `train.py`, so evaluation will run automatically after training.
```bash
# Tested on A100 80GB GPU. You can add "--use_downsample_gaussians" to run on a 3090 24GB GPU.
python train.py -s data/DATASET_NAME/SCAN_ID -o output/DATASET_NAME/SCAN_ID --sfm_config posed --use_view_config --config_view_num 5 --select_inpaint_num 10  --tetra_downsample_ratio 0.25
```
**Note:** The reproduced results may vary due to the randomness inherent in the generative model (See3D), especially in unstructured regions such as ceilings. You may get worse or even better results than those reported in the paper. If the results are worse, simply rerunning the code should produce improved outcomes.

We also provide command for dense-view reconstruction:
```bash
# Tested on 3090 24GB GPU.
python train.py -s data/denseview/scan1 -o output/denseview/scan1 --sfm_config posed --use_view_config --config_view_num 20 --use_downsample_gaussians --tetra_downsample_ratio 0.25 --use_dense_view
```


## Acknowledgements
Some codes are borrowed from [MAtCha](https://github.com/Anttwo/MAtCha), [NeuralPlane](https://github.com/3dv-casia/NeuralPlane), [See3D](https://github.com/baaivision/See3D), [MASt3R-SfM](https://github.com/naver/mast3r), [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields). We thank all the authors for their great work. 


## Citation

```bibtex
@inproceedings{ni2026g4splat,
    title={G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior},
    author={Ni, Junfeng and Chen, Yixin and Yang, Zhifei and Liu, Yu and Lu, Ruijie and Zhu, Song-Chun and Huang, Siyuan},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```
