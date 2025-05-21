import torch
import torchac
import numpy as np

import network_ECC
from Utils import operation
from Utils.data import read_point_cloud, save_point_cloud, get_file_size_in_bits

import numpy as np
from glob import glob
from tqdm import tqdm
import os
import time
import random 

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import warnings
warnings.filterwarnings("ignore")


class Timer:
    def __init__(self):
        self.dict = {}
    
    def start_count(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time()
    
    def end_count(self, label):
        torch.cuda.synchronize()
        self.dict[label] = time.time() - self.dict[label]
    
    def set(self, label, t):
        self.dict[label] = t
    
    def get_sum(self, precision=3, reset=False):
        t = 0
        for key in self.dict.keys():
            t += self.dict[key]
        t = round(t, precision)
        if reset:
            self.dict = {}
        return t
    
class Recoder:
    def __init__(self):
        self.ls = []

    def update(self, value):
        self.ls.append(value)
    
    def get_avg(self, precision=5, reset=False):
        avg_value = round(np.array(self.ls).mean(), precision)
        if reset:
            self.ls = []
        return avg_value

import argparse
parser = argparse.ArgumentParser(
    prog='encode.py',
    description='Compress point clouds.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--input_globs', type=str)
parser.add_argument('--compressed_path', type=str, default='./data/compressed/')
parser.add_argument('--datatype', type=str, help='semantickitti or ford', default="semantickitti")
parser.add_argument('--gpu_id', type=int, help='gpu_id', default=0)
parser.add_argument('--K', type=int, help='$K$.', default=32)
parser.add_argument('--use_oae', action="store_true", help="use oae for skeleton if true else gpcc")
parser.add_argument('--window_size', type=int, help='window size.', default=16)
parser.add_argument('--octree_depth', type=int, help='octree_depth.', default=12)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
from skeleton_encoder import octformer_encode_backbone, tmc_compress, tmc_decompress

if args.datatype == "semantickitti":
    dilated_list = [1, 2, 4]
    model_load_path =  './model/ckpt_ECC_32.pt'
    gpcc_input_scale = 9.5
    pc_scale = 1
elif args.datatype == "ford":
    dilated_list = [1, 1, 2]
    model_load_path =  './model/ckpt_ECC_ford_64.pt'
    gpcc_input_scale = 4
    pc_scale = 1000
else:
    raise Exception("wrong datatype")

#########################################
input_globs = args.input_globs
compressed_path = args.compressed_path
tmc_path = "./tmc3_v29"
local_window_size = args.K
tree_depth = args.octree_depth
batch_size = 64
#########################################

if not os.path.exists(compressed_path):
    os.makedirs(compressed_path)

model = network_ECC.PointModel(channel=64, 
                            bottleneck_channel=16, dilated_list = dilated_list)
model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
model = torch.compile(model)
model = model.cuda().eval()
files = np.array(glob(input_globs, recursive=True))

print("#"*15 + f" local_window_size {local_window_size} " + "#"*15)
with torch.no_grad():
    time_recoder, bpp_recoder = Recoder(), Recoder()
    bone_bpp_recoder = Recoder()
    tq = tqdm(files, ncols=150)
    for file_path in tq:
        timer = Timer()
        pc = read_point_cloud(file_path) / pc_scale
        batch_x = torch.tensor(pc).unsqueeze(0).cuda()
        N = batch_x.shape[1]

        filename_w_ext = os.path.split(file_path)[-1]
        compressed_head_path = os.path.join(compressed_path, filename_w_ext+'.h.bin')
        compressed_skin_path = os.path.join(compressed_path, filename_w_ext+'.s.bin')
        if not args.use_oae:
            compressed_bone_path = os.path.join(compressed_path, filename_w_ext+'.b.bin')
            cache_file = compressed_path + "cache.ply"

        timer.start_count("FPS+KNN")
        bones, local_windows = operation.SamplingAndQuery(batch_x, local_window_size, no_anchor=True)
        timer.end_count("FPS+KNN")

        if args.use_oae:
            timer.start_count("bone encode")
            bone_steam_size, rec_pc, root_octant, min_, max_, db_center, db_extent = \
                    octformer_encode_backbone(bones.detach().cpu().numpy(), depth=tree_depth, \
                                            batch_size=batch_size, save_path=os.path.join(compressed_path, filename_w_ext))
            rec_bones = torch.tensor(rec_pc, dtype=torch.float32).cuda()
            timer.end_count("bone encode")
        else:
            save_point_cloud(bones * gpcc_input_scale, cache_file)
            bone_steam_size, xyz_time = tmc_compress(tmc_path, compressed_path + "cache.ply", compressed_bone_path)
            dec_time = tmc_decompress(tmc_path, compressed_bone_path, cache_file)
            timer.set("bone encode", xyz_time + dec_time)
            rec_bones = torch.tensor(read_point_cloud(cache_file) / gpcc_input_scale, dtype=torch.float32).cuda()

        timer.start_count("reorder")
        cloest_idx = operation.reorder(bones, rec_bones)
        bones, local_windows = bones[cloest_idx], local_windows[cloest_idx]
        timer.end_count("reorder")
        
        timer.start_count("align")
        aligned_windows = operation.AdaptiveAligning(local_windows, rec_bones)
        timer.end_count("align")

        timer.start_count('Feature_Squeeze')
        aligned_windows_in, aligned_windows_out = \
            aligned_windows[:, 0:local_window_size:1,:], aligned_windows[:, 0:local_window_size*2:2,:]
        knn_idx_list = operation.construct_knn_idx_list(aligned_windows_in, args.window_size, dilated_list)
        feature = model.feature_squeeze_in(aligned_windows_in, aligned_windows_in, knn_idx_list) # M, K, C
        max_pooled_feature = torch.max(feature, dim=1, keepdim=False)[0] # M, 1, C

        knn_idx_list = operation.construct_knn_idx_list(aligned_windows_out, args.window_size, dilated_list)
        feature = model.feature_squeeze_out(aligned_windows_out, aligned_windows_out, knn_idx_list) # M, K, C
        max_pooled_feature = torch.concatenate([max_pooled_feature, torch.max(feature, dim=1, keepdim=False)[0]], dim=-1)
        timer.end_count('Feature_Squeeze')

        timer.start_count('Entropy Moddule')
        knn_idx_list = operation.construct_knn_idx_list(rec_bones.unsqueeze(0), 8, [1, 2, 4])
        mu, sigma = model.entropy_Model(rec_bones.unsqueeze(0), knn_idx_list) # M, c * 2
        mu, sigma = mu[0], sigma[0]
        timer.end_count('Entropy Moddule')

        timer.start_count('Encoding')
        compact_fea = model.linear(max_pooled_feature)
        quantized_compact_fea = torch.round(compact_fea)
        ############## 🚩 Arithmetic Encoding ##############
        min_v_value, max_v_value = quantized_compact_fea.min().to(torch.int16), quantized_compact_fea.max().to(torch.int16)
        bytestream = torchac.encode_int16_normalized_cdf(
            operation._convert_to_int_and_normalize(operation.get_cdf_min_max_v(mu-min_v_value, sigma, L=max_v_value-min_v_value+1), needs_normalization=True).cpu(), 
            (quantized_compact_fea-min_v_value).cpu().to(torch.int16)
        )
        timer.end_count('Encoding')
        
        if args.use_oae:
            with open(compressed_head_path, 'wb') as fout:
                fout.write(np.array(local_window_size, dtype=np.uint16).tobytes())
                fout.write(np.array(min_v_value.item(), dtype=np.int16).tobytes())
                fout.write(np.array(max_v_value.item(), dtype=np.int16).tobytes())
                fout.write(np.array(min_, dtype=np.float64).tobytes())
                fout.write(np.array(max_, dtype=np.float64).tobytes())
                fout.write(np.array(db_center[0], dtype=np.float64).tobytes())
                fout.write(np.array(db_center[1], dtype=np.float64).tobytes())
                fout.write(np.array(db_center[2], dtype=np.float64).tobytes())
                fout.write(np.array(db_extent, dtype=np.float64).tobytes())
                fout.write(np.array(root_octant, dtype=np.uint8).tobytes())
        else:
            with open(compressed_head_path, 'wb') as fout:
                fout.write(np.array(local_window_size, dtype=np.uint16).tobytes())
                fout.write(np.array(min_v_value.item(), dtype=np.int16).tobytes())
                fout.write(np.array(max_v_value.item(), dtype=np.int16).tobytes())
                fout.write(np.array(gpcc_input_scale, dtype=np.float64).tobytes())

        with open(compressed_skin_path, 'wb') as fin:
            fin.write(bytestream)

        total_bits = bone_steam_size + get_file_size_in_bits(compressed_skin_path) \
                    + get_file_size_in_bits(compressed_head_path)
        bpp = total_bits / N
        enc_time = timer.get_sum(precision=5)
        time_recoder.update(enc_time)
        bpp_recoder.update(bpp)
        bone_bpp_recoder.update(bone_steam_size/N)
        tq.set_description(f"Bpp: {bpp_recoder.get_avg(precision=3)}, Encode Time: {time_recoder.get_avg(precision=3)}")
    print(f"Bpp: {bpp_recoder.get_avg(precision=3)}, Encode Time: {time_recoder.get_avg(precision=3)}")
