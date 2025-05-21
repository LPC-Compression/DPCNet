import torch
import torch.nn as nn
from pytorch3d.ops.knn import knn_gather
class GraphConv(nn.Module):
    def __init__(self, in_channel, mlps, relu):
        super(GraphConv, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                    mlp_Module = nn.Sequential(
                        nn.Linear(mlps[i], mlps[i+1]),
                        nn.ReLU(inplace=True),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Linear(mlps[i], mlps[i+1]),
                    )
            self.mlp_Modules.append(mlp_Module)
        channel = mlps[-1]

    def forward(self, points):
        """
        Input:
            points: input points position data, [B, ..., N, C]
        Return:
            points: feature data, [B, ..., D]
        """
        for m in self.mlp_Modules:
            points = m(points)
        points = torch.max(points, -2, keepdim=False)[0]
        return points
    
class Folding(nn.Module):
    def __init__(self, in_channel, fold_ratio, out_channel):
        super(Folding, self).__init__()
        self.fold_ratio = fold_ratio
        self.out_channel = out_channel
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, in_channel),
            nn.ReLU(),
            nn.Linear(in_channel, fold_ratio*out_channel),
        )

    def forward(self, fea):
        """
        Input:
            fea: (..., in_channel)
        Output:
            fea: (..., fold_ratio, out_channel)
        """
        output_shape = fea.shape[:-1]+(self.fold_ratio,self.out_channel,)
        fea = self.mlp(fea).reshape(output_shape)
        return fea


class FoldingNetModule(nn.Module):
    def __init__(self, channel, fold_channel, R_max, r):
        super(FoldingNetModule, self).__init__()
        self.R_max = R_max
        self.r = r
        self.folding_base = Folding(in_channel=channel, fold_ratio=R_max, out_channel=fold_channel)

        # self.mlp = nn.Sequential(
        #     nn.Linear(channel * 2, channel),
        #     nn.ReLU(),
        #     nn.Linear(channel, channel),
        #     nn.ReLU(),
        #     nn.Linear(channel, fold_channel),
        # )

        self.folding_pro = Folding(in_channel=channel+fold_channel, fold_ratio=r, out_channel=3)
        self.channel = channel

    def forward(self, skin_features, K):
        """
        Input:
            skin_features: (M, C)
        """
        M = skin_features.shape[0]

        # # # generate fea matrix
        fea = self.folding_base(skin_features) # M, C -> M, R_max, fold_channel
        # get fea: (M, R_max, fold_channel)
        # sampling
        fea = fea[:, torch.randperm(self.R_max)[:K//self.r], :]

        # get fea: (M, K//r, fold_channel)
        # position_encoding = get_positional_encoding(K//self.r, self.channel).unsqueeze(0).repeat(M, 1, 1).to(skin_features.device) # M, K//r, C
        # fea = torch.concatenate([position_encoding, skin_features.unsqueeze(1).repeat(1, K//self.r, 1)], dim=-1) # M, K//r, 2 * C
        # fea = self.mlp(fea) # M, K//r, fold_channel

        # generate xyz
        skin_features = skin_features.unsqueeze(1).repeat((1, fea.shape[1], 1))
        cat_fea = torch.cat((skin_features, fea), dim=-1)
        # get cat_fea: (M, K//r, fold_channel+channel)

        xyz = self.folding_pro(cat_fea)
        # get xyz: (M, K//r, r, 3)
        xyz = xyz.view(M, -1, 3)

        return xyz
    
class AttentionGraphConv(nn.Module):
    def __init__(self, in_channel, mlps, relu):
        super(AttentionGraphConv, self).__init__()

        mlps.insert(0, in_channel)
        self.mlp_Modules = nn.ModuleList()
        for i in range(len(mlps) - 1):
            if relu[i]:
                    mlp_Module = nn.Sequential(
                        nn.Linear(mlps[i], mlps[i+1]),
                        nn.ReLU(inplace=True),
                        )
            else:
                mlp_Module = nn.Sequential(
                    nn.Linear(mlps[i], mlps[i+1]),
                    )
            self.mlp_Modules.append(mlp_Module)
    
        self.attention = nn.Sequential( # M, K, k, 2*C -> M, K, k
            nn.Linear(mlps[-1] * 2, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Flatten(-2),
            nn.Softmax(dim=-1)
        )

        self.act = nn.ELU(inplace=True)

    def forward(self, points):
        """
        Input:
            points: input points position data, [B, ..., N, C]
        Return:
            points: feature data, [B, ..., D]
        """
        for m in self.mlp_Modules:
            points = m(points) # M, K, k, C
        attention_score = self.attention(torch.concatenate([points, points[:, :, 0, :].unsqueeze(-2).expand_as(points)], dim=-1)) # M, K, k
        points = attention_score.unsqueeze(-1) * points
        points = torch.sum(points, dim=-2, keepdim=False)
        return self.act(points)

class ResGCNModule(nn.Module):
    def __init__(self, input_dim, n_layers, embed_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layer = n_layers
        self.GraphConvs = nn.ModuleList()
        self.GraphConvs.append(AttentionGraphConv(input_dim + 9, [embed_dim//4, embed_dim//2, embed_dim], [True, True, True]))
        for i in range(1, n_layers):
            self.GraphConvs.append(AttentionGraphConv(embed_dim + 9, [embed_dim], [True]))
        self.output_emb = nn.Sequential(
            nn.Linear(self.n_layer * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
    
    def forward(self, x, xyz, knn_idx_list): # 1, M, 3  n * 1, M, k 
        assert self.n_layer == len(knn_idx_list), f"knn_idx_list 长度 {len(knn_idx_list)} 与 设置的参数 {self.n_layer}不匹配"
        feature_list = []
        for i in range(self.n_layer):
            feature = knn_gather(x, knn_idx_list[i]) # M K k C
            position = knn_gather(xyz, knn_idx_list[i]) # M K k 3
            substract = position - xyz.unsqueeze(-2) # M K k 3
            distance = torch.sqrt(substract * substract) # M K k 1
            feature = torch.concatenate([feature, position, substract, distance], dim=-1) # 1, M, k, C+7
            if i == 0:
                x = self.GraphConvs[i](feature) # # 1, M, k, C+7 -> 1, M, C
            else:
                x = self.GraphConvs[i](feature) + x
            feature_list.append(x)
        feature = torch.concatenate(feature_list, dim=-1) # 1, M, C * (self.n_layer)
        feature = self.output_emb(feature) # 1, M, C
        return feature

class EntropyModule(nn.Module):
    def __init__(self, n_layers, embed_dim, bottleneck_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layer = n_layers
        self.bottleneck_channel = bottleneck_channel
        self.GraphConvs = nn.ModuleList()
        self.GraphConvs.append(AttentionGraphConv(3 + 9, [embed_dim//4, embed_dim//2, embed_dim], [True, True, True]))
        for i in range(1, n_layers):
            self.GraphConvs.append(AttentionGraphConv(embed_dim + 9, [embed_dim], [True]))
        self.output_emb = nn.Sequential(
            nn.Linear(self.n_layer * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, bottleneck_channel * 2)
        )
    
    def forward(self, input, knn_idx_list): # 1, M, 3  n * 1, M, k 
        assert self.n_layer == len(knn_idx_list), f"knn_idx_list 长度 {len(knn_idx_list)} 与 设置的参数 {self.n_layer}不匹配"
        feature_list = []
        x = input
        for i in range(self.n_layer):
            feature = knn_gather(x, knn_idx_list[i]) # 1, M, k, C
            position = knn_gather(input, knn_idx_list[i]) # 1, M, k, 3
            substract = position - input.unsqueeze(-2) # 1, M, k, 3  - 1, M, 1, 3
            distance = torch.sqrt(substract * substract)
            feature = torch.concatenate([feature, position, substract, distance], dim=-1) # 1, M, k, C+9
            if i == 0:
                x = self.GraphConvs[i](feature) # M, K, C
            else:
                x = self.GraphConvs[i](feature) + x
            feature_list.append(x)
        feature = torch.concatenate(feature_list, dim=-1) # M, K, C * (self.n_layer)
        mu_sigma = self.output_emb(feature) # M, K, C
        mu, sigma = mu_sigma[:, :, :self.bottleneck_channel], torch.exp(mu_sigma[:, :, self.bottleneck_channel:])
        return mu, sigma