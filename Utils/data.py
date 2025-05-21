import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import torch
import os
from torch.utils.data import DataLoader,Dataset
from prefetch_generator import BackgroundGenerator
import math


def read_point_cloud(filepath):
    pc = PyntCloud.from_file(filepath)
    pc = np.array(pc.points, dtype=np.float32)[:, :3]
    return pc

def save_point_cloud(pc, path):
    if torch.is_tensor(pc):
        pc = pc.detach().cpu().numpy()
    df = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(path)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

class PrefetchDataLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=5)
    
class SemanticKITTIDataset(Dataset):
    def __init__(self, data_root, split="train", ignore_index=-1, sample_ratio=-1):
        super(SemanticKITTIDataset, self).__init__()
        self.ignore_index = ignore_index
        self.split = split
        self.data_root = data_root
        self.sample_ratio = sample_ratio
        assert sample_ratio == -1 or 0< sample_ratio < 1
        self.data_list = self.get_data_list()
    
    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError
        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            if self.sample_ratio > 0:
                total_nums = len(seq_files)
                sampled_count = max(2, math.ceil(total_nums * self.sample_ratio))
                new_seq_files = [seq_files[0], seq_files[total_nums-1]]
                step = (total_nums - 1) / (sampled_count - 1)
                for i in range(1, sampled_count-1):
                    new_seq_files.insert(-1, seq_files[math.ceil(i * step)])
                seq_files = new_seq_files
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        return coord.astype(np.float32)

class CustomDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc = read_point_cloud(self.files[idx])
        return pc
    
