# Towards Multi-Layered 3D Garments Animation
\[[Paper](https://arxiv.org/abs/2305.10418)\] | \[[Project](https://mmlab-ntu.github.io/project/layersnet/index.html)\] | \[[Dataset](https://github.com/ftbabi/D-LAYERS_ICCV2023.git)\]

This is the official repository of "Towards Multi-Layered 3D Garments Animation, ICCV 2023".

**Authors**: Yidi Shao, [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/),  and [Bo Dai](http://daibo.info/).

**Acknowedgement**: This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP20221-0011) and Shanghai AI Laboratory.


**Feel free to ask questions. I am currently working on some other stuff but will try my best to reply. Please don't hesitate to star!**

## News
- 4 Aug, 2023: Codes released

## Table of Content
1. [Video Demos](#video-demos)
2. [Dataset](#dataset)
3. [Code](#code)
4. [Citations](#citations)

## Video Demos
![](imgs/demo.gif)

Please refer to our [project page](https://mmlab-ntu.github.io/project/layersnet/index.html) for more details.


## Dataset
Please follow [this repo](https://github.com/ftbabi/D-LAYERS_ICCV2023.git) to download and prepare the dataset.


## Code
Codes are tested on Ubuntu 18 and cuda 11.3.
We train our model with 4 V100.


### Installation
1. Create a conda environment
```
conda create -n LayersNet python=3.11
conda activate LayersNet
```
2. install pytorch (using `conda install` is very slow, so using pip3)
```
pip3 install torch torchvision torchaudio
```
3. install and build `mmcv`
```
git clone git@github.com:kai-lan/mmcv.git --branch 1.x
cd mmcv
MMCV_WITH_OPS=1 pip install -e .
```
4. Install other dependent packages
```
pip3 install h5py pyrender trimesh numpy==1.23.1 tqdm plotly scipy chumpy matplotlib
```
5. Clone and install this repo forked from the origin
```
git clone git@github.com:kai-lan/LayersNet_ICCV2023.git

cd LayersNet_ICCV2023
pip3 install -v -e .
```

### Dataset Preparation
1. Datasets have been prepared for you, and you just need to create a link inside your `LayersNet_ICCV2023` folder:
```
ln -s /data2/D-Layers/D-LAYERS___D-LAYERS data
```
2. Data have been preprocessed for you, and are saved in `/data2/D-Layers/D-LAYERS___D-LAYERS/generated_data`. The following commends are for reference only, and you __DO NOT__ need to run them.
```
# Prepare the static information
python tools/preprocessing_data.py configs/layersnet/base/ep1.py --work_dir output --dataset train --type static

# Prepare the dynamic information, e.g., velocity
python tools/preprocessing_data.py configs/layersnet/base/ep1.py --work_dir output --dataset train --type dynamic
```

### Testing
1. Rollout the results of one sequence, e.g., sequence `00396` here:
```
python tools/test.py configs/layersnet/base/test.py data/ckpt.pth --work_dir output --show-dir output --show-options rollout=396
```
2. Save quantitative results to a json file, e.g., `eval.json` here:
```
python tools/test.py configs/layersnet/base/test.py data/ckpt.pth --out output/00396/eval.json
```

### Visualization
To visualize the output, here take 64th frame from sequence `00396` as example, please use the following command
TODO: this can only visualize one frame at a time. How to generate a video from it?
```
python tools/visualization.py configs/layersnet/base/test.py --work_dir output --seq 396 --frame 64
```

TODO: update training
---
### Training
Train on multiple GPUs
```
sh tools/dist_seq.sh PATH/TO/CONFIG/DIR/ NUM_GPUS PATH/TO/WORK/DIR/ --seed 0
```

## Citations
```
@inproceedings{shao2023layersnet,
  author = {Shao, Yidi and Loy, Chen Change and Dai, Bo},
  title = {Towards Multi-Layered {3D} Garments Animation},
  booktitle = {ICCV},
  year = {2023}
}
```
