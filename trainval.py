# ---------------------------------------------------------------------------- #
# Train and validate the model
# Modified from GeoTrans, with support for both 3D point cloud and image data
# ---------------------------------------------------------------------------- #
import time
import torch.optim as optim
import os.path as osp
from src.backbone_3d.engine.epoch_based_trainer import EpochBasedTrainer
import argparse
from easydict import EasyDict as edict
from utils.common import load_yaml, dir_exist
from src.dataloaders.data_loader import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
from src.backbone_2d_patchnet.patchnet import PatchNetAutoencoder


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)

        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler

        model_img = PatchNetAutoencoder(256, normalize=True)
        self.model_img = model_img

        model = create_model(cfg, self.model_img).cuda()
        model = self.register_model(model)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        self.optimizer_img = optim.Adam(model_img.parameters(), lr=cfg.optim.lr_img, weight_decay=cfg.optim.weight_decay)
        self.scheduler_img = optim.lr_scheduler.StepLR(self.optimizer_img, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def postprocess_cfg(cfg):
    cfg.root_dir = osp.dirname(osp.abspath(__file__))
    cfg.output_dir = osp.join(cfg.root_dir, cfg.dir.output_folder_name, cfg.dir.exp_name)
    cfg.snapshot_dir = osp.join(cfg.output_dir, "snapshots")
    cfg.log_dir = osp.join(cfg.output_dir, "logs")
    cfg.event_dir = osp.join(cfg.output_dir, "events")
    # cfg.feature_dir = osp.join(cfg.output_dir, "features")
    # cfg.registration_dir = osp.join(cfg.output_dir, "registration")

    # dir_exist(cfg.output_dir, sub_folders=["snapshots", "logs", "events", "features", "registration"])
    dir_exist(cfg.output_dir, sub_folders=["snapshots", "logs", "events"])

    cfg.kpconv.init_radius = cfg.kpconv.base_radius * cfg.kpconv.init_voxel_size
    cfg.kpconv.init_sigma = cfg.kpconv.base_sigma * cfg.kpconv.init_voxel_size
    return cfg


def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/train/3DMatch.yaml', help='Path to config file.')
    args = parser.parse_args()
    cfg = load_yaml(args.config, keep_sub_directory=True)
    cfg = edict(cfg)

    cfg = postprocess_cfg(cfg)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
