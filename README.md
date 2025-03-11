# All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting

[![arXiv](https://img.shields.io/badge/arXiv-KeySS-green.svg?style=plastic)](https://arxiv.org/abs/2503.07191) [![HuggingFace](https://img.shields.io/badge/HuggingFace-PretrainedMipnerf360Models-blue.svg?style=plastic)](https://huggingface.co/jojojohn/mipnerf360_pretrained) <br>

This repository contains the official authors implementation associated with the paper ["All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting."](https://arxiv.org/abs/2410.18775)

> **All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting**<br>
> Yan Ren, Shilin Lu, Adams Wai-Kin Kong <br>
>
>**Abstract**: <br>
*Recent advances in 3D Gaussian Splatting (3DGS) have revolutionized scene reconstruction, opening new possibilities for 3D steganography by hiding 3D secrets within 3D covers. The key challenge in steganography is ensuring imperceptibility while maintaining high-fidelity reconstruction. However, existing methods often suffer from detectability risks and utilize only suboptimal 3DGS features, limiting their full potential.
We propose a novel end-to-end key-secured 3D steganography framework (KeySS) that jointly optimizes a 3DGS model and a key-secured decoder for secret reconstruction. Our approach reveals that Gaussian features contribute unequally to secret hiding. The framework incorporates a key-controllable mechanism enabling multi-secret hiding and unauthorized access prevention, while systematically exploring optimal feature update to balance fidelity and security. To rigorously evaluate steganographic imperceptibility beyond conventional 2D metrics, we introduce 3D-Sinkhorn distance analysis, which quantifies distributional differences between original and steganographic Gaussian parameters in the representation space.
Extensive experiments demonstrate that our method achieves state-of-the-art performance in both cover and secret reconstruction while maintaining high security levels, advancing the field of 3D steganography.*

---

![Teaser image](assets/flowtry.png)

---

## Contents
  - [Environment Installation](#setup)
  - [Dataset and Pretrained Model](#dataset-and-pretrained-model)
  - [Running](#running)
  - [Evaluation](#evaluation)
  - [Acknowledgement](#evaluation)
  - [Citation](#citation)

<br>

## Environment Installation
To install KeySS, please run the following commands. Note that our setup was tested on a server with a Quadro RTX 6000 GPU and CUDA 11.8:
```shell
git clone https://github.com/RY-Paper/KeySS.git 
conda create -n keyss python=3.8 -y
conda activate keyss
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install plyfile
pip install tqdm scipy wandb opencv-python scikit-learn lpips
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install transformers
pip install -U "huggingface_hub[cli]"
```

## Dataset and Pretrained Model
The dataset needed consists of 9 scenes from MipNeRF360 dataset ([available here](https://jonbarron.info/mipnerf360/)) and 1 scene from Deep Blending dataset ([available here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)). These scenes are used as cover and secret scenes in our experiments.

You can download the datasets using the following commands:
```
cd KeySS
mkdir data
wget -c http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip 360_v2.zip -d mipnerf360
wget -c https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
unzip 360_extra_scenes.zip -d mipnerf360
wget -c https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip 
unzip tandt_db.zip
```

Pretrained MipNeRF360 3DGS models can be downloaded using the following commands:
```
cd KeySS
huggingface-cli download jojojohn/mipnerf360_pretrained --repo-type=model --local-dir output
cd output
unzip keyss_gt_mipnerf360.zip 
```
Please follow the dataset structure below to prepare the datasets and pretrained models in the source path location:
```
<location>
|---data
|   |---mipnerf360 #datasets
|     |---bicycle
|     |---...
|   |---db #datasets
|     |---playroom
|---output
    |---mipnerf360 #pretrained models
    |---single_secret
    |---multiple_secert
        
```


## Running

To train for single/multiple secret hiding, simply see the details in run_train.sh

```shell
bash run_train.sh
```

## Evaluation

To test for single/multiple secret hiding and 3D-Sinkhorn, simply see the details in run_test.sh

```shell
bash run_test.sh
```

## Acknowledgement
We would like to thank the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping/tree/main) and [CLIP](https://github.com/openai/CLIP) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.


## Citation
If you find our work helpful, please consider citing:
```bibtex
@article{keyss2025,
  title={All That Glitters Is Not Gold: Key-Secured 3D Secrets within 3D Gaussian Splatting},
  author={Ren, Yan and Lu, Shilin and Kong, Adams Wai-Kin},
  journal={arXiv preprint arXiv:2503.07191},
  year={2025}
}
```



