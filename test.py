# ---------------------------------------------------------------------------- #
# Test script for 3D point cloud matching using both 3D point cloud and image data
# Modified from GeoTrans
# ---------------------------------------------------------------------------- #
import argparse
import os.path as osp
import time
import numpy as np
from src.backbone_3d.engine.single_tester import SingleTester
from utils.torch import release_cuda
from utils.common import get_log_string, dir_exist, load_yaml
from src.dataloaders.data_loader import test_data_loader
from model import create_model
from loss import Evaluator
from src.backbone_2d_patchnet.patchnet import PatchNetAutoencoder
from easydict import EasyDict as edict


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        print(f'Data loader created: {loading_time:.3f}s collapsed.')
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        # model = create_model(cfg).cuda()
        # add patch-based image model part
        model_img = PatchNetAutoencoder(256, normalize=True)
        self.model_img = model_img

        model = create_model(cfg, self.model_img).cuda()

        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator(cfg).cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
        dir_exist(self.output_dir)

    def test_step(self, iteration, data_dict):
        # output flops and model parameters
        # from thop import profile
        # flops, params = profile(self.model, inputs=(data_dict,))

        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(output_dict, data_dict)
        return result_dict

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_frame = data_dict['ref_frame']
        src_frame = data_dict['src_frame']
        message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
        message += ', ' + get_log_string(result_dict=result_dict)
        message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict['scene_name']
        ref_id = data_dict['ref_frame']
        src_id = data_dict['src_frame']

        dir_exist(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f'{ref_id}_{src_id}.npz')
        np.savez_compressed(
            file_name,
            ref_points=release_cuda(output_dict['ref_points']),
            src_points=release_cuda(output_dict['src_points']),
            # ref_points_f=release_cuda(output_dict['ref_points_f']),
            # src_points_f=release_cuda(output_dict['src_points_f']),
            ref_points_c=release_cuda(output_dict['ref_points_c']),
            src_points_c=release_cuda(output_dict['src_points_c']),
            # ref_feats_c=release_cuda(output_dict['ref_feats_c']),
            # src_feats_c=release_cuda(output_dict['src_feats_c']),
            # ref_feats_c_fuse=release_cuda(output_dict['ref_feats_c_fuse']),
            # src_feats_c_fuse=release_cuda(output_dict['src_feats_c_fuse']),
            # ref_feats_f=release_cuda(output_dict['ref_feats_f']),
            # src_feats_f=release_cuda(output_dict['src_feats_f']),
            ref_node_corr_indices=release_cuda(output_dict['ref_node_corr_indices']),
            src_node_corr_indices=release_cuda(output_dict['src_node_corr_indices']),
            ref_corr_points=release_cuda(output_dict['ref_corr_points']),
            src_corr_points=release_cuda(output_dict['src_corr_points']),
            corr_scores=release_cuda(output_dict['corr_scores']),
            gt_node_corr_indices=release_cuda(output_dict['gt_node_corr_indices']),
            gt_node_corr_overlaps=release_cuda(output_dict['gt_node_corr_overlaps']),
            estimated_transform=release_cuda(output_dict['estimated_transform']),
            transform=release_cuda(data_dict['transform']),
            overlap=data_dict['overlap'],
        )


def postprocess_cfg(cfg):
    cfg.root_dir = osp.dirname(osp.abspath(__file__))
    cfg.output_dir = osp.join(cfg.root_dir, cfg.dir.output_folder_name, cfg.dir.exp_name)
    # cfg.snapshot_dir = osp.join(cfg.output_dir, "snapshots")
    cfg.log_dir = osp.join(cfg.output_dir, "logs")
    # cfg.event_dir = osp.join(cfg.output_dir, "events")
    cfg.feature_dir = osp.join(cfg.output_dir, "features")
    cfg.registration_dir = osp.join(cfg.output_dir, "registration")

    # dir_exist(cfg.output_dir, sub_folders=["snapshots", "logs", "events", "features", "registration"])
    dir_exist(cfg.output_dir, sub_folders=["logs", "features", "registration"])

    cfg.kpconv.init_radius = cfg.kpconv.base_radius * cfg.kpconv.init_voxel_size
    cfg.kpconv.init_sigma = cfg.kpconv.base_sigma * cfg.kpconv.init_voxel_size
    return cfg


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='3DMatch_planar',
                        choices=['3DMatch', '3DLoMatch', '3DMatch_planar', '3DLoMatch_planar',
                                 'IndoorLRS', 'IndoorLRS_planar',
                                 'ScanNetpp_test', 'ScanNetpp_test_planar', 'val'],
                        help='test benchmark')
    return parser


def main():
    # start_time_0 = time.time()
    # load configs
    if make_parser().parse_args().benchmark in ['3DMatch', '3DLoMatch', '3DMatch_planar', '3DLoMatch_planar']:
        cfg = load_yaml(osp.join("./configs/test/3DMatch.yaml"), keep_sub_directory=True)
    elif make_parser().parse_args().benchmark in ['IndoorLRS', 'IndoorLRS_planar']:
        cfg = load_yaml(osp.join("./configs/test/IndoorLRS.yaml"), keep_sub_directory=True)
    elif make_parser().parse_args().benchmark in ['ScanNetpp_test', 'ScanNetpp_test_planar']:
        cfg = load_yaml(osp.join("./configs/test/ScanNetpp.yaml"), keep_sub_directory=True)
    cfg = edict(cfg)

    cfg = postprocess_cfg(cfg)
    tester = Tester(cfg)
    tester.run()
    # total_time = time.time() - start_time_0
    # print(f'Total time: {total_time:.3f}s.')


if __name__ == '__main__':
    main()
