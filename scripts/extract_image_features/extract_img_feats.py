# ---------------------------------------------------------------------------- #
# Extract pixel-wise image features from multiview RGB images
# ---------------------------------------------------------------------------- #
import argparse
import time
import os
from datetime import datetime

import torch
from easydict import EasyDict as edict

from backbones import build_backbone
from utils.common import load_yaml
from utils.o3d_tools import *

from datasets.extractor_3dmatch import ThreeDMatchDataset_Process
from datasets.extractor_indoorlrs import IndoorLRSDataset_Process
from datasets.extractor_scannetpp import ScanNetppDataset_Process


def load_state_with_same_shape(model, weights):
    """
    Load weights into model, filtering only those with matching shape.
    """
    model_state = model.state_dict()
    return {
        k[9:]: v for k, v in weights.items()
        if k[9:] in model_state and v.size() == model_state[k[9:]].size()
    }


def resume_checkpoint(backbone2d, checkpoint_dir='models/checkpoint.pth'):
    """
    Resume model weights from a checkpoint.
    """
    if os.path.isfile(checkpoint_dir):
        print('===> Loading existing checkpoint')
        state = torch.load(checkpoint_dir, weights_only=True, map_location='cpu')
        matched_weights = load_state_with_same_shape(backbone2d, state['model'])
        backbone2d.load_state_dict(matched_weights, strict=False)
        del state
    else:
        print(f'===> No checkpoint found at {checkpoint_dir}. Please check the path.')
    return backbone2d


def load_pretrain_model(model_name, dim, checkpoint_dir):
    """
    Initialize and load pretrained backbone.
    """
    backbone2d = build_backbone(model_name, dim, pretrained=True)
    return resume_checkpoint(backbone2d, checkpoint_dir)


def main():
    parser = argparse.ArgumentParser("Extract pixel-wise image features from multiview RGB images")
    parser.add_argument('--config', type=str, default='./configs/extract_scannetpp.yaml',
                        help='Path to config file.')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    config = edict(cfg)

    start_time = time.strftime('%Y%m%d_%H:%M:%S')

    config.backbone2d = load_pretrain_model('Res50UNet', config.img_feats_dim, config.pretrain_root)
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset-specific feature extraction
    if config.dataset == '3DMatch' and config.benchmark_name in ['train', 'val', '3DMatch', '3DLoMatch']:
        ThreeDMatchDataset_Process(config)
    elif config.dataset == 'IndoorLRS' and config.benchmark_name in ['IndoorLRS', 'IndoorLRS_planar']:
        IndoorLRSDataset_Process(config)
    elif config.dataset == 'ScanNetpp' and config.benchmark_name in ['ScanNetpp_test', 'ScanNetpp_test_planar']:
        ScanNetppDataset_Process(config)
    else:
        raise ValueError(f'Unsupported dataset type: {config.dataset}')

    end_time = time.strftime('%Y%m%d_%H:%M:%S')
    with open(os.path.join(config.output_root, 'main_time_feat_extract.txt'), 'a') as f:
        f.write(f'mode: {config.dataset, config.benchmark_name}\n')
        f.write(f'start_time: {start_time}\n')
        f.write(f'end_time: {end_time}\n')
        f.write('total time:\n')
        f.write(str(datetime.strptime(end_time, '%Y%m%d_%H:%M:%S') -
                    datetime.strptime(start_time, '%Y%m%d_%H:%M:%S')) + '\n')


if __name__ == '__main__':
    main()
