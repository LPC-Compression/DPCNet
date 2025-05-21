import os
import argparse
import subprocess

import numpy as np
from glob import glob
from tqdm import tqdm

from multiprocessing import Pool

def parse_psnr_output(output_str):
    c = output_str.splitlines()
    for i in range(len(c)):
        if c[i].startswith('3.'):
            d1 = float(c[i+2].split(' ')[-1])
            try:
                d2 = float(c[i+4].split(' ')[-1])
            except Exception as e:
                d2 = 0.
            break
    return d1, d2

from scipy.spatial import KDTree
def distChamfer(f1, f2, scale=1.0):
    f1 /= scale
    f2 /= scale
    tree = KDTree(f1, compact_nodes=False)
    d1, _ = tree.query(f2, k=1, workers=-1, eps=0)
    tree = KDTree(f2, compact_nodes=False)
    d2, _ = tree.query(f1, k=1, workers=-1, eps=0)
    return max(d1.mean(), d2.mean())

from Utils.data import read_point_cloud
def process(input_f):
    filename_w_ext = os.path.split(input_f)[-1]
    dec_f = os.path.join(args.decompressed_path, filename_w_ext+'.bin.ply')

    cmd = f'./pc_error \
    -a {input_f} -b {dec_f} \
    -r {peak}'
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    d1_psnr, d2_psnr = parse_psnr_output(output.decode("utf-8"))

    origin_pc = read_point_cloud(input_f)
    rec_pc = read_point_cloud(dec_f)
    cd = distChamfer(origin_pc, rec_pc, scale=charmfer_scale)
    
    return np.array([filename_w_ext, d1_psnr, d2_psnr, cd])


parser = argparse.ArgumentParser(
    prog='eval_PSNR.py',
    description='Eval Geometry PSNR.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_globs', type=str, help='Glob pattern to load point clouds.')
parser.add_argument('--decompressed_path', type=str, help='Path to save decompressed files.', default='./data/decompressed/')
parser.add_argument('--datatype', type=str, help='semantickitti or ford', default="semantickitti")
args = parser.parse_args()

if args.datatype == "semantickitti":
    peak = "59.70"
    charmfer_scale = 0.01
elif args.datatype == "ford":
    peak = "30000"
    charmfer_scale = 10
else:
    raise Exception("wrong datatype")

files = np.array(glob(args.input_globs))

files = np.sort(files)

f_len = len(files)
with Pool(16) as p:
    arr = list(tqdm(p.imap(process, files), total=f_len, ncols=150))

arr = np.array(arr)
fnames, p2pPSNRs, p2plainPSNRs = arr[:, 0], arr[:, 1].astype(float), arr[:, 2].astype(float)

cds = arr[:, 3].astype(float)

stri = f'Avg. D1 PSNR: {p2pPSNRs.mean()}'
print(stri)
stri = f'Avg. D2 PSNR: {p2plainPSNRs.mean()}'
print(stri)
stri = f'Avg. CD: {cds.mean()}'
print(stri)