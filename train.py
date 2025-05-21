import os
import random
import argparse

import numpy as np
from glob import glob
from datetime import datetime

import torch
from pytorch3d.loss import chamfer_distance

from Utils.data import SemanticKITTIDataset, PrefetchDataLoader, CustomDataset
import network_ECC
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--datatype', type=str, help='semantickitti or ford', default="semantickitti")
parser.add_argument('--gpu_id', type=int, help='gpu_id', default=0)
parser.add_argument('--model_save_folder', type=str, help='Directory where to save trained models.', default=f'./model/ckpt_kitti.pt')
parser.add_argument('--train_glob', type=str, help='Glob pattern to load point clouds.', default='./Ford_full/Ford_01_q_1mm/*.ply')
parser.add_argument('--K', type=int, help='$K$.', default=32)
parser.add_argument('--dilated_window_size', type=int, help='Dilated window size $k$.', default=8)
parser.add_argument('--channel', type=int, help='Network channel.', default=64)
parser.add_argument('--bottleneck_channel', type=int, help='Bottleneck channel.', default=16)
parser.add_argument('--λ_R', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-2)
parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion tradeoff at x steps.', default=5000)
parser.add_argument('--batch_size', type=int, help='Batch size (must be 1).', default=1)
parser.add_argument('--lr', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, help='Decays the learning rate at x step.', default=[70000, 120000])
parser.add_argument('--max_step', type=int, help='Train up to this number of steps.', default=140000)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

if args.datatype == "semantickitti":
    dilated_list = [1, 2, 4]
    dataset = SemanticKITTIDataset(data_root=args.train_glob, split="train")
elif args.datatype == "ford":
    dilated_list = [1, 1, 2]
    files = glob(args.train_glob)
    dataset = CustomDataset(files)
else:
    raise Exception("wrong datatype")

model = network_ECC.PointModel(channel=args.channel, 
                          bottleneck_channel=args.bottleneck_channel, dilated_list = dilated_list)

model = model.cuda().train()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loader = PrefetchDataLoader(
    dataset = dataset,
    batch_size = args.batch_size,
    shuffle = True
)

global_step = 0
cd_recoder, bpp_recoder, loss_recoder = Recoder(), Recoder(), Recoder()

progress_bar = tqdm(total=140000, ncols=150)

for epoch in range(1, 9999):
    print(datetime.now())
    for batch_x in loader:
        
        batch_x = batch_x.cuda()

        if random.random() < 0.1 and global_step > 100000:
            K = random.randint(args.K + 1, 3600)
        else:
            K = args.K

        optimizer.zero_grad()
        rec_batch_x, bitrate = model(batch_x, K)
        # Get Loss
        chamfer_dist, _ = chamfer_distance(rec_batch_x, batch_x)
        loss = chamfer_dist
        if global_step > args.rate_loss_enable_step:
            loss = loss +  args.λ_R * bitrate
        loss.backward()
        global_step += 1
        progress_bar.update(1)
  
        optimizer.step()

        cd_recoder.update(chamfer_dist.item() * 100)
        bpp_recoder.update(bitrate.item())
        loss_recoder.update(loss.item())

        if global_step % 500 == 0:
            progress_bar.set_description(f'Step:{global_step} Ave CD: {cd_recoder.dump_avg(precision=3)} ' + \
                                         f'Bpp: {bpp_recoder.dump_avg(precision=3)} Loss: {loss_recoder.dump_avg(precision=3)}')

            # save model
            torch.save(model.state_dict(), args.model_save_folder)
        
        # Learning Rate Decay
        if global_step in args.lr_decay_steps:
            args.lr = args.lr * args.lr_decay
            for g in optimizer.param_groups:
                g['lr'] = args.lr
            print(f'Learning rate decay triggered at step {global_step}, LR is setting to{args.lr}.')

        if global_step > args.max_step:
            break
    
    if global_step > args.max_step:
        break
