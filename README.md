# RadarMask
Here is the official code repository for RadarMask. The code will be open-sourced upon acceptance. Currently, the dataset section is available.

# Overview
RadarMask is an end-to-end method for panoptic segmentation and tracking in the radar domain, requiring no post-processing.

# Dependencies
The main dependencies of the project are the following:
```
python=3.8
cuda==12.1
```
(we test on python=3.8.19, pytorch==2.3.1, cuda==12.1)
You can set up a conda environment as follows:
```
conda create --name radarmask python=3.8
conda activate radarmask
```
```
# conda&CUDA 12.1:
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# pip&CUDA 12.1:
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
```
git clone https://github.com/yb-guo/RadarMask.git 
cd RadarMask

conda install --yes --file requirements.txt
# Install this package by running in the root directory of this repo:
pip3 install -U -e .
```

Install [SparseTransformer](https://github.com/dvlab-research/SparseTransformer)

# Data:
Download the [RadarScenes](https://radar-scenes.com/) and set path in radarscenes.py.
The datasets directory structure should like this:
```
.
└── data
│   ├── sequence_1
│   │   └── camera
│   │   ├── radar_data.h5
│   │   └── scenes.json
│   ├── sequence_2
│   │   └── camera
│   │   └── camera
│   │   ├── radar_data.h5
│   │   └── scenes.json    
│   ├── sequence_3
│   │   └── camera
│   │   ├── radar_data.h5
│   │   └── scenes.json
......
│   └── sequences.json
├── License.md
└── Readme.md

```

