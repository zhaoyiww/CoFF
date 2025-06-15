from src.dataloaders.datasets.threedmatch import ThreeDMatchPairDataset
from src.dataloaders.datasets.indoorlrs import IndoorLRSDataset
from src.dataloaders.datasets.scannetpp import ScanNetppDataset

from src.dataloaders.datasets_utils import (
    registration_collate_fn_stack_mode,
    registration_collate_fn_stack_mode_3dmatch,
    # specific data preprocessing for indooelrs, scannetpp
    registration_collate_fn_stack_mode_indoorlrs,
    registration_collate_fn_stack_mode_scannetpp,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode_3dmatch,
        cfg.kpconv.num_stages,
        cfg.kpconv.init_voxel_size,
        cfg.kpconv.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode_3dmatch,
        cfg.kpconv.num_stages,
        cfg.kpconv.init_voxel_size,
        cfg.kpconv.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.val.point_limit,
        use_augmentation=False,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode_3dmatch,
        cfg.kpconv.num_stages,
        cfg.kpconv.init_voxel_size,
        cfg.kpconv.init_radius,
        neighbor_limits,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    # train_dataset = ThreeDMatchPairDataset(
    #     cfg.data.dataset_root,
    #     'train',
    #     point_limit=cfg.train.point_limit,
    #     use_augmentation=cfg.train.use_augmentation,
    #     augmentation_noise=cfg.train.augmentation_noise,
    #     augmentation_rotation=cfg.train.augmentation_rotation,
    # )
    # neighbor_limits = calibrate_neighbors_stack_mode(
    #     train_dataset,
    #     registration_collate_fn_stack_mode,
    #     cfg.backbone.num_stages,
    #     cfg.backbone.init_voxel_size,
    #     cfg.backbone.init_radius,
    # )

    # fix the neighbor limits for different datasets, avoiding reloading the training set
    neighbor_limits = [53, 34, 34, 38]
    # use a default setting to get rid of reloading 3DMatch training set during inference,
    # the inference result may be slightly different to the previous one

    if benchmark.startswith('3DMatch') or benchmark.startswith('3DLoMatch'):
        test_dataset = ThreeDMatchPairDataset(
            cfg.data.dataset_root,
            benchmark,
            point_limit=cfg.test.point_limit,
            use_augmentation=False,
        )
        test_loader = build_dataloader_stack_mode(
            test_dataset,
            registration_collate_fn_stack_mode_3dmatch,
            cfg.kpconv.num_stages,
            cfg.kpconv.init_voxel_size,
            cfg.kpconv.init_radius,
            neighbor_limits,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
        )
    elif benchmark.startswith('IndoorLRS'):
        test_dataset = IndoorLRSDataset(
            cfg.data.dataset_root,
            benchmark,
            point_limit=cfg.test.point_limit,
            use_augmentation=False,
        )
        test_loader = build_dataloader_stack_mode(
            test_dataset,
            registration_collate_fn_stack_mode_indoorlrs,
            cfg.kpconv.num_stages,
            cfg.kpconv.init_voxel_size,
            cfg.kpconv.init_radius,
            neighbor_limits,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
        )
    elif benchmark.startswith('ScanNetpp'):
        test_dataset = ScanNetppDataset(
            cfg.data.dataset_root,
            benchmark,
            point_limit=cfg.test.point_limit,
            use_augmentation=False,
        )
        # a = registration_collate_fn_stack_mode
        ############
        # # skip the first k samples during debugging
        # from torch.utils.data import Subset
        # # 458 has large memory
        # subset = Subset(test_dataset, range(459, len(test_dataset)))
        # ############

        test_loader = build_dataloader_stack_mode(
            test_dataset,
            # subset,
            registration_collate_fn_stack_mode_scannetpp,
            cfg.kpconv.num_stages,
            cfg.kpconv.init_voxel_size,
            cfg.kpconv.init_radius,
            neighbor_limits,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
        )

    # test_dataset = ThreeDMatchPairDataset(
    #     cfg.data.dataset_root,
    #     benchmark,
    #     point_limit=cfg.test.point_limit,
    #     use_augmentation=False,
    # )
    # test_loader = build_dataloader_stack_mode(
    #     test_dataset,
    #     registration_collate_fn_stack_mode,
    #     cfg.backbone.num_stages,
    #     cfg.backbone.init_voxel_size,
    #     cfg.backbone.init_radius,
    #     neighbor_limits,
    #     batch_size=cfg.test.batch_size,
    #     num_workers=cfg.test.num_workers,
    #     shuffle=False,
    # )

    return test_loader, neighbor_limits
