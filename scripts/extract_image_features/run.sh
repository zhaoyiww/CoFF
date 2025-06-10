#!/bin/bash

python extract_img_feats.py --config "./configs/extract_3dmatch.yaml"
python extract_img_feats.py --config "./configs/extract_indoorlrs.yaml"
python extract_img_feats.py --config "./configs/extract_scannetpp.yaml"