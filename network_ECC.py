import torch
import torch.nn as nn

import Utils.operation as op
from Utils.nn import FoldingNetModule, ResGCNModule, EntropyModule


class PointModel(nn.Module):
    def __init__(self, channel, bottleneck_channel, dilated_list=[1, 2, 4]):
        super(PointModel, self).__init__()
        self.dilated_list = dilated_list

        self.feature_squeeze_in = ResGCNModule(input_dim=3, n_layers=3, embed_dim=channel, output_dim=channel)
        self.feature_squeeze_out = ResGCNModule(input_dim=3, n_layers=3, embed_dim=channel, output_dim=channel)
        self.linear = nn.Linear(channel*2, bottleneck_channel)

        self.entropy_Model = EntropyModule(n_layers=3, embed_dim=channel, bottleneck_channel=bottleneck_channel)

        self.fea_stretch = ResGCNModule(input_dim=bottleneck_channel, n_layers=3, embed_dim=channel, output_dim=channel)
        self.point_generator = FoldingNetModule(channel=channel, fold_channel=8, R_max=256, r=4)

    def forward(self, batch_x, K):
        N = batch_x.shape[1]

        bones, local_windows = op.SamplingAndQuery(batch_x, K)
        aligned_windows = op.AdaptiveAligning(local_windows, bones) # M, K, 3

        aligned_windows_in, aligned_windows_out = aligned_windows[:, 0:K:1,:], aligned_windows[:, 0:K*2:2,:]
        knn_idx_list = op.construct_knn_idx_list(aligned_windows_in, 16, self.dilated_list)
        feature = self.feature_squeeze_in(aligned_windows_in, aligned_windows_in, knn_idx_list) # M, K, C
        max_pooled_feature = torch.max(feature, dim=1, keepdim=False)[0] # M, 1, C

        knn_idx_list = op.construct_knn_idx_list(aligned_windows_out, 16, self.dilated_list)
        feature = self.feature_squeeze_out(aligned_windows_out, aligned_windows_out, knn_idx_list) # M, K, C
        max_pooled_feature = torch.concatenate([max_pooled_feature, torch.max(feature, dim=1, keepdim=False)[0]], dim=-1)
        feature = self.linear(max_pooled_feature)

        # quantized_compact_fea = feature + (torch.round(feature) - feature).detach()
        quantized_compact_fea = feature + torch.nn.init.uniform_(torch.zeros_like(feature), -0.5, 0.5)

        knn_idx_list = op.construct_knn_idx_list(bones.unsqueeze(0), 8, [1, 2, 4])
        mu, sigma = self.entropy_Model(bones.unsqueeze(0), knn_idx_list) # M, c * 2
        bitrate, _ = op.feature_probs_based_mu_sigma(quantized_compact_fea, mu.squeeze(0), sigma.squeeze(0))
        bitrate = bitrate / N

        feature = self.fea_stretch(quantized_compact_fea.unsqueeze(0), bones.unsqueeze(0), knn_idx_list).squeeze(0) # 1, M, C
        rec_windows = self.point_generator(feature, K)
        rec_windows = op.InverseAligning(rec_windows, bones)
        rec_batch_x = rec_windows.view(1, -1, 3)
        return rec_batch_x, bitrate