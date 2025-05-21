import torch
import torchac
import numpy as np

import network_ECC
from Utils import operation
from Utils.data import save_point_cloud, read_point_cloud

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
parser.add_argument('--compressed_path', type=str, default='./data/compressed/')
parser.add_argument('--decompressed_path', type=str, default='./data/decompressed/')
parser.add_argument('--gpu_id', type=int, help='gpu_id', default=0)
parser.add_argument('--datatype', type=str, help='semantickitti or ford', default="semantickitti")
parser.add_argument('--use_oae', action="store_true", help="use oae for skeleton if true else gpcc")
parser.add_argument('--window_size', type=int, help='window size.', default=16)
parser.add_argument('--octree_depth', type=int, help='octree_depth.', default=12)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
from skeleton_encoder import tmc_decompress, octformer_decode_backbone

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
compressed_path = args.compressed_path
decompressed_path = args.decompressed_path
tree_depth = args.octree_depth
tmc_path = "./tmc3_v29"
batch_size = 64
#########################################

if not os.path.exists(decompressed_path):
    os.makedirs(decompressed_path)

model = network_ECC.PointModel(channel=64, 
                            bottleneck_channel=16, dilated_list = dilated_list)
model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
model = torch.compile(model)
model = model.cuda().eval()

compressed_head_path_ls = list(glob(os.path.join(compressed_path, '*.h.bin')))

with torch.no_grad():
    time_recoder = Recoder()
    tq = tqdm(compressed_head_path_ls, ncols=150)
    for compressed_head_path in tq:
        timer = Timer()
        
        filename_w_ext = os.path.split(compressed_head_path[:-6])[-1]
        compressed_head_path = os.path.join(compressed_path, filename_w_ext+'.h.bin')
        compressed_skin_path = os.path.join(compressed_path, filename_w_ext+'.s.bin')
        if not args.use_oae:
            compressed_bone_path = os.path.join(compressed_path, filename_w_ext+'.b.bin')
            cache_file = compressed_path + "cache.ply"

        if args.use_oae:
            with open(compressed_head_path, 'rb') as fin:
                local_window_size = np.frombuffer(fin.read(2), dtype=np.uint16)[0]
                min_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]
                max_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]
                min_ = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                max_ = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                rec_db_center0 = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                rec_db_center1 = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                rec_db_center2 = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                rec_db_extent = np.frombuffer(fin.read(8), dtype=np.float64)[0]
                rec_octant = np.frombuffer(fin.read(1), dtype=np.uint8)[0]
        else:
            with open(compressed_head_path, 'rb') as fin:
                local_window_size = np.frombuffer(fin.read(2), dtype=np.uint16)[0]
                min_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]
                max_v_value = np.frombuffer(fin.read(2), dtype=np.int16)[0]
                gpcc_input_scale = np.frombuffer(fin.read(8), dtype=np.float64)[0]

        if args.use_oae:
            timer.start_count("bone_dec")
            rec_bones = octformer_decode_backbone(save_path=os.path.join(compressed_path, filename_w_ext), rec_octant=rec_octant,\
                                                rec_db_center0=rec_db_center0, rec_db_center1=rec_db_center1, rec_db_center2=rec_db_center2,\
                                                    rec_db_extent=rec_db_extent, min_=min_, max_=max_, depth=tree_depth, seq_size=512, batch_size=batch_size)
            rec_bones = torch.tensor(rec_bones, dtype=torch.float32).cuda()
            timer.end_count("bone_dec")
        else:
            bone_dec_time = tmc_decompress(tmc_path, compressed_bone_path, cache_file)
            timer.set("bone_dec", bone_dec_time)
            rec_bones = torch.tensor(read_point_cloud(cache_file) / gpcc_input_scale, dtype=torch.float32).cuda()

        timer.start_count('Entropy Moddule')
        knn_idx_list = operation.construct_knn_idx_list(rec_bones.unsqueeze(0), 8, [1, 2, 4])
        mu, sigma = model.entropy_Model(rec_bones.unsqueeze(0), knn_idx_list) # M, c * 2
        mu, sigma = mu[0], sigma[0]
        timer.end_count('Entropy Moddule')

        with open(compressed_skin_path, 'rb') as fin:
            bytestream = fin.read()

        timer.start_count('Decoding')
        quantized_compact_fea = torchac.decode_int16_normalized_cdf(
            operation._convert_to_int_and_normalize(operation.get_cdf_min_max_v(mu-min_v_value, sigma, L=max_v_value-min_v_value+1), needs_normalization=True).cpu(), 
            bytestream
        ) + min_v_value
        quantized_compact_fea = quantized_compact_fea.float().cuda()
        timer.end_count('Decoding')

        timer.start_count('DWUS')
        feature = model.fea_stretch(quantized_compact_fea.unsqueeze(0), rec_bones.unsqueeze(0), knn_idx_list).squeeze(0)
        rec_windows = model.point_generator(feature, local_window_size)
        rec_windows = operation.InverseAligning(rec_windows, rec_bones)
        rec_batch_x = rec_windows.view(1, -1, 3)
        timer.end_count('DWUS') # 

        decompressed_result_path = os.path.join(decompressed_path, filename_w_ext+'.bin.ply')
        save_point_cloud(rec_batch_x[0] * pc_scale, decompressed_result_path)

        time_recoder.update(timer.get_sum(precision=7))

        tq.set_description(f"Decode Time {time_recoder.get_avg(precision=3)}")
    print(f"Decode Time {time_recoder.get_avg(precision=3)}")