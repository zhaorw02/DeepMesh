<p align="center">
  <h3 align="center"><strong>DeepMesh: Auto-Regressive Artist-Mesh Creation<br>With Reinforcement Learning</strong></h3>

<p align="center">
    <a href="https://zhaorw02.github.io/">Ruowen Zhao</a><sup>1*</sup>,
    <a href="https://jamesyjl.github.io/">Junliang Ye</a><sup>1*</sup>,
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>1*</sup>,<br>
    <a href="">Guangce Liu</a><sup>2</sup>,
    <a href="https://buaacyw.github.io/">Yiwen Chen</a><sup>3</sup>,
    <a href="https://yikaiw.github.io/">Yikai Wang</a><sup>1</sup>
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>1,2†</sup>,
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    <sup>†</sup>Corresponding authors.
    <br>
    <sup>1</sup>Tsinghua University,
    <sup>2</sup>ShengShu,
    <br>
    <sup>3</sup>S-Lab, Nanyang Technological University,
</p>


<div align="center">

<a href='https://arxiv.org/abs/2406.10163'><img src='https://img.shields.io/badge/arXiv-2406.10163-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://zhaorw02.github.io/DeepMesh/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/buaacyw/MeshAnything/blob/master/LICENSE.txt'><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/Yiwen-ntu/MeshAnything/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;

</div>


<p align="center">
    <img src="asserts/teaser.png" alt="Demo" width="1024px" />
</p>


## Release
- [3/20] 🔥🔥We released the 0.5B version of **DeepMesh**.

## Contents
- [Release](#release)
- [Installation](#installation)
- [Usage](#usage)
- [Important Notes](#important-notes)
- [Todo](#todo)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with A100, A800 and A6000.
1. Clone our repo and create conda environment
```
git clone https://github.com/zhaorw02/DeepMesh.git && cd DeepMesh
conda env create -f environment.yaml
conda activate deepmesh
```
2. Install the pretrained model weight
```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download zzzrw/DeepMesh --local-dir ./
```

## Usage
### Command line inference
```
# Note: if you want to use your own point cloud, please make sure the normal is included.
# The file format should be a .npy file with shape (N, 6), where N is the number of points. The first 3 columns are the coordinates, and the last 3 columns are the normal.

# Generate all obj/ply in your folder
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=12345 sample.py \
    --model_path "your_model_path" \
    --steps 90000 \
    --input_path examples \
    --output_path mesh_output \
    --repeat_num 4 \
    --temperature 0.5 \

# Generate the specified obj/ply in your folder
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=22345.py \
    --model_path "your_model_path" \
    --steps 90000 \
    --input_path examples \
    --output_path mesh_output \
    --repeat_num 4 \
    --uid_list "wand1.obj,wand2.obj,wand3.ply" \
    --temperature 0.5 \
```
## Important Notes
- Please refer to our [project_page](https://zhaorw02.github.io/DeepMesh/) for more examples.
## Todo

The repo is still being under construction, thanks for your patience. 
- [ ] Release of pre-training code  ( truncted sliding training ).
- [ ] Release of post-training code ( DPO ).
- [ ] Release of larger model ( 1b version ).

## Acknowledgement
Our code is based on these wonderful repos:
* **[BPT](https://github.com/whaohan/bpt)**
* **[LLaMA-Mesh](https://github.com/nv-tlabs/LLaMa-Mesh)**
* [Meshanything](https://github.com/buaacyw/MeshAnythingV2/tree/main)
* [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
* [transformers](https://github.com/huggingface/transformers)

## BibTeX
```
```
