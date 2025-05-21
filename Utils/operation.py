import torch
from pytorch3d.ops.knn import knn_points
import math


from pytorch3d.ops.knn import knn_points
def construct_knn_idx_list(points, k, dilated_list): # M, K, 3
    max_k = k * max(dilated_list)
    _, idx, _ = knn_points(points, points, K=max_k, return_sorted=True) # M, K, k
    knn_idx_list = []
    for d in dilated_list:
        knn_idx_list.append(idx[:, :,0:k*d:d])
    return knn_idx_list

from pytorch3d.ops.sample_farthest_points import sample_farthest_points
def SamplingAndQuery(batch_x, K, no_anchor=False, ratio=1.5):
    _, N, _ = batch_x.shape
    M = N*2//K
    # Sampling
    if N < 10000 or no_anchor:
        bones = sample_farthest_points(batch_x, K=M)[0] # (1, M, 3)
    else:
        sample_anchor = batch_x.clone()[:, torch.randperm(N)[:M*16], :]
        bones = sample_farthest_points(sample_anchor, K=M)[0] # (1, M, 3)
    # Query
    _, _, local_windows = knn_points(bones, batch_x, K=int(K*ratio), return_nn=True)
    bones, local_windows = bones[0], local_windows[0]
    return bones, local_windows

def reorder(points, ref_points):
    '''
    Input:
        points: 
        ref_points: 
    '''

    dist = torch.cdist(points.cpu(), ref_points.cpu())
    cloest_idx = torch.argmin(dist, dim=0).cuda()

    return cloest_idx

def get_self_cd(pos):
    '''
    input:
        pos: (B, N, 3)
    output:
        dist: (B, N)
    '''
    dist = knn_points(pos, pos, K=2, return_nn=False).dists[:,:,1]
    dist = torch.sqrt(dist)
    return dist

def feature_probs_based_mu_sigma(feature, mu, sigma):
    sigma = sigma.clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, probs


def get_cdf_min_max_v(mu, sigma, L):
    M, d = sigma.shape
    mu = mu.unsqueeze(-1).repeat(1, 1, L)
    sigma = sigma.unsqueeze(-1).repeat(1, 1, L).clamp(1e-10, 1e10)
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    flag = torch.arange(0, L).to(sigma.device).view(1, 1, L).repeat((M, d, 1))
    cdf = gaussian.cdf(flag + 0.5)

    spatial_dimensions = cdf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=cdf.dtype, device=cdf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    # print(cdf_with_0.shape)
    return cdf_with_0


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
  """
  From torchac
  """
  Lp = cdf_float.shape[-1]
  factor = torch.tensor(
    2, dtype=torch.float32, device=cdf_float.device).pow_(16)
  new_max_value = factor
  if needs_normalization:
    new_max_value = new_max_value - (Lp - 1)
  cdf_float = cdf_float.mul(new_max_value)
  cdf_float = cdf_float.round()
  cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
  if needs_normalization:
    r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
    cdf.add_(r)
  return cdf

def AdaptiveAligning(local_windows, bones):
    n_local_windows = local_windows - bones.unsqueeze(-2)  # (M, K, 3)
    sampled_self_dist = get_self_cd(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows / sampled_self_dist # -> (M, K, 3)
    return n_local_windows


def InverseAligning(n_local_windows, bones):
    sampled_self_dist = get_self_cd(bones.unsqueeze(0))[0].view(-1, 1, 1) # -> (M, 1, 1)
    sampled_self_dist = sampled_self_dist[sampled_self_dist[:, 0, 0] != 0]
    sampled_self_dist = sampled_self_dist.mean()
    n_local_windows = n_local_windows * sampled_self_dist # -> (M, K, 3)
    local_windows = n_local_windows + bones.unsqueeze(-2)
    return local_windows