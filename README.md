<p align="center">

  <h1 align="center">CoFF: Cross-Modal Feature Fusion for Robust Point Cloud Registration with Ambiguous Geometry</h1>
  <p align="center">
    <a href="https://github.com/zhaoyiww/CoFF"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/zhaoyiww/CoFF"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://www.sciencedirect.com/science/article/pii/S0924271625001935"><img src="https://img.shields.io/badge/Paper-pdf-<COLOR>.svg?style=flat-square" /></a>
    <a href="https://github.com/zhaoyiww/CoFF/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" /></a>
  </p>  

  <p align="center">
    <a href="https://zhaoyiww.github.io/"><strong>Zhaoyi Wang</strong></a>
    ·
    <a href="https://shengyuh.github.io/"><strong>Shengyu Huang</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/jemil-avers-butt.html"><strong>Jemil Avers Butt</strong></a>
    ·
    <a href="https://github.com/yuanjua"><strong>Yuanzhou Cai</strong></a>
    ·
    <a href="https://github.com/mvarga1989"><strong>Matej Varga</strong></a>
    ·
    <a href="https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html"><strong>Andreas Wieser</strong></a>
  </p>
  <p align="center"><a href="https://ethz.ch/en.html"><strong>ETH Zürich</strong></a>

  <div align="center"></div>
</p>

<div style="text-align: center;">
<img src="assets/method_overview_isprs_coff_25.jpg" alt="Logo" style="width:100%; height:auto;">
</div>

This repository provides the official Pytorch implementation of the paper: [Cross-Modal Feature Fusion for Robust Point Cloud Registration with Ambiguous Geometry](https://www.sciencedirect.com/science/article/pii/S0924271625001935), published in the ISPRS Journal.

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>📚 Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#code-base-overview">Code base overview</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data preparation</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#todo-list">TODO list</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## 🧠 Abstract

<details>
    <summary>[Abstract (click to expand)]</summary>

Point cloud registration has seen significant advancements with the application of deep learning techniques. However, existing approaches often overlook the potential of integrating radiometric information from RGB images. This limitation reduces their effectiveness in aligning point clouds pairs, especially in regions where geometric data alone is insufficient. When used effectively, radiometric information can enhance the registration process by providing context that is missing from purely geometric data. In this paper, we propose CoFF, a novel Cross-modal Feature Fusion method that utilizes both point cloud geometry and RGB images for pairwise point cloud registration. Assuming that the co-registration between point clouds and RGB images is available, CoFF explicitly addresses the challenges where geometric information alone is unclear, such as in regions with symmetric similarity or planar structures, through a two-stage fusion of 3D point cloud features and 2D image features. It incorporates a cross-modal feature fusion module that assigns pixel-wise image features to 3D input point clouds to enhance learned 3D point features, and integrates patch-wise image features with superpoint features to improve the quality of coarse matching. This is followed by a coarse-to-fine matching module that accurately establishes correspondences using the fused features. We extensively evaluate CoFF on four common datasets: 3DMatch, 3DLoMatch, IndoorLRS, and the recently released ScanNet++ datasets. In addition, we assess CoFF on specific subset datasets containing geometrically ambiguous cases. Our experimental results demonstrate that CoFF achieves state-of-the-art registration performance across all benchmarks, including remarkable registration recalls of 95.9% and 81.6% on the widely-used 3DMatch and 3DLoMatch datasets, respectively. CoFF is particularly effective in scenarios with challenging geometric features, provided that RGB images are available and that the overlapping regions exhibit sufficient texture in the RGB images.
</details>

## 🧱 Code base overview

Our code builds upon [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) framework, a 3D geometry-based point cloud registration method. The following key extensions are introduced:

- 🔀 Integration of 2D RGB and 3D geometric features at both point-pixel and patch-patch levels;
- 🧩 Modules for image preprocessing, offline image feature extraction, and feature lifting;
- 🔍 Evaluation scripts for additional datasets: [IndoorLRS](http://redwood-data.org/indoor_lidar_rgbd/) and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/).

We recommend users refer to [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) for more details on the core architecture.

### Registration recall (RR) performance across datasets

|     Method \ RR |    3DM    |   3DLM    |  IndoorLRS  |  ScanNet++  |   3DM_P   |  3DLM_P   |  IndoorLRS_P   |  ScanNet++_P   |
|----------------|:---------:|:---------:|:-----------:|:-----------:|:---------:|:---------:|:--------------:|:--------------:|
|        GeoTrans |   91.5    |   74.0    |    96.2     |    73.4     |   76.4    |   59.5    |      91.2      |      37.6      |
| **CoFF (ours)** | **95.9**  | **81.6**  |  **97.1**   |  **78.7**   | **90.5**  | **70.4**  |    **94.2**    |    **56.0**    |

> **Note**: 3DM = 3DMatch, 3DLM = 3DLoMatch; `_P` indicates the 'Planar' subset of each dataset.

## 🛠️ Installation

## 🗂️ Data preparation

CoFF is evaluated on the following datasets:

- 3DMatch and 3DLoMatch
- IndoorLRS
- ScanNet++

Please follow the [instructions for data preprocessing](./scripts/README.md) to prepare the data and metadata.

## 🚀 Run

We provide our pre-trained model weight for direct testing and evaluation, as well as weights for baseline methods. All weights can be downloaded [here](https://drive.google.com/drive/folders/1pBet7ZwG8aj6H3kezeOppvF_ac-rJ8IU?usp=sharing). Note that all models are trained on the 3DMatch training set and evaluated on the 3DMatch test set, 3DLoMatch, IndoorLRS, and ScanNet++ datasets.

To use CoFF with the pre-trained model, place the downloaded weights in the `./weights/` folder.

### Test and evaluate on datasets

1. Modify the config file for the target dataset under the `./configs/test/` folder. Set the appropriate paths and parameters. Default configurations for all four datasets are provided.

2. To test the model and extract features from the test dataset, run:

```bash
bash test.sh
```

This will extract features from the test dataset. To evaluate the registration performance, run:

```bash
bash eval.sh
```

For IndoorLRS and ScanNet++ datasets, additionally run 'benchmark_indoorlrs.py' and 'benchmark_scannetpp.py' in the './scripts/' folder to complete the evaluation.

### 🏋️ Train your own model

To train your own model, modify the config file under the './configs/train/' folder and set the appropriate paths and parameters. The default configurations for the 3DMatch training set is provided.

Then, run the following command to train the model:

```bash
python trainval.py --config ./configs/train/3DMatch.yaml
```

## ✅ TODO list
- [x] Upload data preprocessing scripts.
- [x] Upload complete training and evaluation code.
- [x] Upload our pretrained model weight.
- [ ] Upload our preprocessed data. 

## 🤝 Acknowledgements

We sincerely thank the following works and use parts of their official implementations:

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer/tree/main) for the 3D backbone (our code is also heavily borrowed from it);
- [LCD](https://github.com/hkust-vgd/lcd.git) for the 2D patch-based backbone;
- [Pri3D](https://github.com/Sekunde/Pri3D) and [PCR-CG](https://github.com/Gardlin/PCR-CG.git) for the 2D pixel-based backbone;
- [3DMatch](https://3dmatch.cs.princeton.edu/) for the 3DMatch dataset;
- [Predator](https://github.com/prs-eth/OverlapPredator.git) for the 3DLoMatch dataset;
- [Color-ICP](http://redwood-data.org/indoor_lidar_rgbd/) for the IndoorLRS dataset;
- [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) for the ScanNet++ dataset.


## 🧾 Citation

If you find our code or paper useful, please cite:

```bibtex
@article{Wang2025CoFF,
    title = {Cross-modal feature fusion for robust point cloud registration with ambiguous geometry},
    author = {Wang, Zhaoyi and Huang, Shengyu and Butt, Jemil Avers and Cai, Yuanzhou and Varga, Matej and Wieser, Andreas},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {227},
    pages = {31--47},
    year = {2025},
}
```
