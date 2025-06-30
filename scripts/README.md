## Overview

This document provides an overview of data preprocessing pipelines for four benchmark datasets. It includes the following key components:
- **Image selection**: For memory-constrained settings, we provide scripts to select a limited number of images (e.g., 3 or 5) per point cloud fragment.
- **Preprocessing**: Scripts for generating point cloud fragments (if not provided in the official dataset, e.g., ScanNet++), and metadata files such as `gt.log` and `.pkl`.
- **Subset creation**: Tools for constructing geometrically planar subsets.
- **Image feature extraction**: Offline scripts to extract pixel-wise image features

Preprocessed data can be downloaded from [CoFF preprocessed data](https://www.research-collection.ethz.ch/handle/20.500.11850/742804) (note: **ScanNet++ is not included due to licensing restrictions and must be requested individually**). If you'd like to preprocess the datasets yourself, or prepare your own data, follow the steps below.

## Preprocessing instructions

### 3DMatch and 3DLoMatch

- Download the original RGB-D reconstruction data from the [official 3DMatch website](https://3dmatch.cs.princeton.edu/#rgbd-reconstruction-datasets).
- The preprocessed "geometry" directory can be downloaded from [Predator's 3DMatch release](https://share.phys.ethz.ch/~gseg/pairwise_reg/).

To extract dense image features, please refer to [extract_img_feats.py](./extract_image_features/extract_img_feats.py). The directory structure should resemble:

```
./3DMatch
├── geometry
└── image
```

### IndoorLRS

- Download RGB-D sequences, fragments, and camera poses from the [official IndoorLRS website](http://redwood-data.org/indoor_lidar_rgbd/download.html).
- Use script ['preprocess_indoorlrs.py'](./preprocess/preprocess_indoorlrs.py) to generate metadata files (e.g., `IndoorLRS.pkl` and `benchmarks/`).

```
./IndoorLRS
├── geometry
└── image
```

### ScanNet++

- Download raw RGB and depth images from the [ScanNet++ official website](https://kaldir.vc.in.tum.de/scannetpp/).
- As fragments are not provided, use [`generate_pcd_fragments_scannetpp.py`](./preprocess/generate_pcd_fragments_scannetpp.py) to generate them. We segment every 20 consecutive frames into a point cloud fragment.
- Use script [`preprocess_scannetpp.py`](./preprocess/preprocess_scannetpp.py) to generate metadata files (e.g., `ScanNetpp.pkl` and `benchmarks/`).
    

The directory structure for pixel-wise image feature extraction:

```
./ScanNetpp
├── fragments
├── image
│   └── <scene_id>
│       ├── rgb/
│       ├── depth/
│       └── pose_intrinsic_imu.json
└── metadata
    └── split
        └── test_scannetpp.txt

```

## Final data structure for training and testing

We train CoFF on the **3DMatch train set** and directly evaluate on all test sets. For better performance, training on individual datasets (e.g., ScanNet++) is also supported. The recommended directory layout is:

```
./3DMatch
├── data
│   ├── train
│   └── test
├── image
│   ├── <scene>
│   │   ├── seq-01/
│   │   ├── seq-02/
│   │   └── ...
├── metadata

./IndoorLRS
├── data
│   └── test
├── image
│   ├── apartment/
│   ├── camera-intrinsics.txt
│   ├── pose_apartment.log
│   └── ...
├── metadata

./ScanNetpp
├── data
│   └── test
├── image
├── metadata
```

**Alternatively**, you can directly use our processed metadata files available at `./data/[dataset_name]/metadata/`.

Note: The `.pkl` metadata format follows the convention of **GeoTransformer**. For compatibility with **Predator**, a conversion script [`pkl_Predator2GeoTrans.py`](./create_subsets/pkl_Predator2GeoTrans.py) is provided.
