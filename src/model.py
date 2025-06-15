# ---------------------------------------------------------------------------- #
# Source from GeoTrans, modified for 3D point cloud and image data
# ---------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.backbone_3d.modules.ops import point_to_node_partition, index_select
from src.backbone_3d.modules.registration import get_node_correspondences
from src.backbone_3d.modules.sinkhorn import LearnableLogOptimalTransport
from src.backbone_3d.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone_3d.kpconv import KPConvFPN


class CoFF(nn.Module):
    def __init__(self, cfg, model_img):
        super(CoFF, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        # normal
        self.kpconv = KPConvFPN(
            # 1, 256, 64
            cfg.kpconv.input_dim,
            cfg.kpconv.output_dim,
            cfg.kpconv.init_dim,
            # 15, 2.5 * 0.025, 2 * 0.025, 32
            cfg.kpconv.kernel_size,
            cfg.kpconv.init_radius,
            cfg.kpconv.init_sigma,
            cfg.kpconv.group_norm,
        )

        # coarse, middle operation
        self.transformer = GeometricTransformer(
            cfg.geotrans.input_dim,
            cfg.geotrans.output_dim,
            cfg.geotrans.hidden_dim,
            cfg.geotrans.num_heads,
            cfg.geotrans.blocks,
            cfg.geotrans.sigma_d,
            cfg.geotrans.sigma_a,
            cfg.geotrans.angle_k,
            reduction_a=cfg.geotrans.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # PatchNet autoencoder
        self.patchnet = model_img

        # fuse feats using a simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(512, 256, bias=True)
            # nn.BatchNorm1d(256),
            # nn.ReLU()
        )

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def score_estimate(self, ref_feats, src_feats, len_ref):
        ref_feats_c_img, src_feats_c_img = (ref_feats.unsqueeze(0).transpose(1, 2),
                                            src_feats.unsqueeze(0).transpose(1, 2))
        feats_c_img = torch.cat([ref_feats_c_img, src_feats_c_img], dim=-1)
        len_ref_c_img = len_ref
        feats_c_img = self.proj_gnn(feats_c_img)
        scores_c_img = self.proj_score(feats_c_img)

        feats_gnn_norm_img = F.normalize(feats_c_img, p=2, dim=1).squeeze(0).transpose(0, 1)  # [N, C]
        scores_c_raw_img = scores_c_img.squeeze(0).transpose(0, 1)  # [N, 1]
        ref_feats_gnn_img, src_feats_gnn_img = feats_gnn_norm_img[:len_ref_c_img], feats_gnn_norm_img[len_ref_c_img:]
        inner_products_img = torch.matmul(ref_feats_gnn_img, src_feats_gnn_img.transpose(0, 1))

        ref_scores_c_img, src_scores_c_img = scores_c_raw_img[:len_ref_c_img], scores_c_raw_img[len_ref_c_img:]
        temperature_img = torch.exp(self.epsilon) + 0.03
        s1_img = torch.matmul(F.softmax(inner_products_img / temperature_img, dim=1), src_scores_c_img)
        s2_img = torch.matmul(F.softmax(inner_products_img.transpose(0, 1) / temperature_img, dim=1), ref_scores_c_img)
        scores_saliency_img = torch.cat((s1_img, s2_img), dim=0)
        sigmoid = nn.Sigmoid()
        # safe guard our score
        scores_saliency_img = torch.clamp(sigmoid(scores_saliency_img.view(-1)), min=0, max=1)
        scores_saliency_img = self.regular_score(scores_saliency_img)
        ref_saliency_img = scores_saliency_img[:len_ref_c_img]
        src_saliency_img = scores_saliency_img[len_ref_c_img:]

        return ref_saliency_img, src_saliency_img

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        # input feats, all ones; [src_n + tgt_n, 1]
        feats = data_dict['features'].detach()
        # add 128D pixel-wise img features
        dim = 128
        feats = feats.repeat(1, dim + 1)
        ref_length = data_dict['ref_img_feats'].shape[0]
        feats[:ref_length, :dim] = data_dict['ref_img_feats']
        feats[ref_length:, :dim] = data_dict['src_img_feats']

        # gt
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        # raw coordinates
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        # downsampled, [0, 1, 2, 3], coarse: 1 --
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f

        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        # self.num_points_in_patch: 64
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        # coarse
        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)

        # each coarse pts select a 64x3 neighbor
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        # coarse
        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        # encoder: 1 --> 4; feats: [40k, 1], 1-->64-->128-->256-->512-->1024
        # inputs: feats: [40k, 1]
        # outputs: feats_list : list of 3, downsampled [10k, 256], [2.6k, 512], [700, 1024]
        # coarse feats: [700, 1024]
        # fine feats: [10k, 256]
        feats_list = self.kpconv(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        # input: [700, 1024]
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        # geometric transformer, dim. of outputs does not change
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching, dimension of 256
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        #########################################################
        # patch_feats and decode patch img
        img_patch_idx_ref = data_dict['img_patch_idx'][:data_dict['img_patch_length'][0]]
        img_patch_idx_src = data_dict['img_patch_idx'][data_dict['img_patch_length'][0]:] - data_dict['lengths'][3][0]

        img_patch_ref = data_dict['img_patch'][:data_dict['img_patch_length'][0]]
        img_patch_src = data_dict['img_patch'][data_dict['img_patch_length'][0]:]

        # gt indices that existed in img patch index list
        indices_col1 = [idx for idx, val in enumerate(gt_node_corr_indices[:, 0]) if val in img_patch_idx_ref]
        indices_col2 = [idx for idx, val in enumerate(gt_node_corr_indices[:, 1]) if val in img_patch_idx_src]
        # # img_patch_idx_gt_ref = [elem for elem in img_patch_idx_ref if elem in gt_node_corr_indices[:, 0]]
        # # img_patch_idx_gt_src = [elem for elem in img_patch_idx_src if elem in gt_node_corr_indices[:, 1]]

        # select the indices of gt if both ref and src patch exist
        common_indices = list(set(indices_col1) & set(indices_col2))

        # randomly select gt pairs
        if self.training:
            sel = 100
        else:
            sel = len(common_indices)

        if len(common_indices) > sel:
            idx = np.random.choice(len(common_indices), sel, replace=False)
            selected_gt_pair = gt_node_corr_indices[np.array(common_indices)[idx]]
            selected_overlaps = gt_node_corr_overlaps[np.array(common_indices)[idx]]
        else:
            sel = len(common_indices)
            selected_gt_pair = gt_node_corr_indices[common_indices]
            selected_overlaps = gt_node_corr_overlaps[common_indices]

        img_patch_idx_ref_2, img_patch_idx_src_2 = [], []
        for elem in range(selected_gt_pair.shape[0]):
            index_idx_ref = torch.where(img_patch_idx_ref == selected_gt_pair[elem, 0])[0]
            index_idx_src = torch.where(img_patch_idx_src == selected_gt_pair[elem, 1])[0]
            # if len(index) > 0:
            img_patch_idx_ref_2.append(index_idx_ref.item())
            img_patch_idx_src_2.append(index_idx_src.item())

        img_patch_ref_2 = np.array(img_patch_ref)[img_patch_idx_ref_2]
        img_patch_src_2 = np.array(img_patch_src)[img_patch_idx_src_2]

        img_patch_ref_2 = torch.tensor(img_patch_ref_2).to('cuda')
        img_patch_src_2 = torch.tensor(img_patch_src_2).to('cuda')

        img_patch_ref_recon, img_patch_ref_feats = self.patchnet(img_patch_ref_2)
        img_patch_src_recon, img_patch_src_feats = self.patchnet(img_patch_src_2)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img_patch_ref_2[10].cpu())
        # plt.show()

        # plt.figure()
        # plt.imshow(img_patch_ref_recon[10].detach().cpu())
        # plt.show()

        output_dict['img_patch_ref_input'] = img_patch_ref_2
        output_dict['img_patch_src_input'] = img_patch_src_2

        output_dict['img_patch_ref_recon'] = img_patch_ref_recon
        output_dict['img_patch_ref_feats'] = img_patch_ref_feats
        output_dict['img_patch_src_recon'] = img_patch_src_recon
        output_dict['img_patch_src_feats'] = img_patch_src_feats

        output_dict['selected_gt_pair'] = selected_gt_pair
        output_dict['selected_overlaps'] = selected_overlaps

        output_dict['gt_transform'] = transform

        # fuse feats if scores_img is higher than scores_pcd
        ref_feats_pcd_sel = ref_feats_c_norm[selected_gt_pair[:, 0]]
        src_feats_pcd_sel = src_feats_c_norm[selected_gt_pair[:, 1]]

        ref_feats_c_fuse = torch.cat([ref_feats_pcd_sel, img_patch_ref_feats], dim=1)
        src_feats_c_fuse = torch.cat([src_feats_pcd_sel, img_patch_src_feats], dim=1)
        ref_feats_c_fuse = self.mlp(ref_feats_c_fuse)
        src_feats_c_fuse = self.mlp(src_feats_c_fuse)

        output_dict['ref_feats_c_fuse'] = ref_feats_c_fuse
        output_dict['src_feats_c_fuse'] = src_feats_c_fuse
        ##################################################################

        # fuse feats for all
        ref_feats_c_norm[selected_gt_pair[:, 0]] = ref_feats_c_fuse
        src_feats_c_norm[selected_gt_pair[:, 1]] = src_feats_c_fuse

        #########################################################
        # dont understand the goal
        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            # coarse matching index
            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                # feats are not used?
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            # fine matching pts
            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        return output_dict


def create_model(config, model_img):
    model = CoFF(config, model_img)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
