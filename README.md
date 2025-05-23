# DPCNet
This is the official code for our paper "DPCNet: Dual-Path Compression Network with Multi-Scale Geometric Context for LiDAR Point Cloud."

## Visualization Results
The visualization results of DPCNet point cloud compression on SemanticKITTI and Ford datasets:  
<video src="https://LPC-Compression.github.io/DPCNet/Video.mp4" controls width="600"></video>

## Installation

Directly run the `install.sh` script:
```
./install.sh
```

## Data
### Downloading
#### KITTI dataset

Please refer to the official [website of KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php), Then download the 80GB dataset "odometry data set (velodyne laser data, 80 GB)".

#### Ford dataset

You need to be a member of MPEG first. The Ford dataset is provided on their [website](https://mpegfs.int-evry.fr/mpegcontent/ws-mpegcontent/MPEG-I).

## Train DPCNet
python train.py --datatype semantickitti  --gpu_id 0 --model_save_folder ./model/kitti.pt --K 32 --train_glob ./datasets/SemanticKITTIDataset/



python train.py --datatype ford  --gpu_id 0 --model_save_folder ./model/ford.pt --K 64 --train_glob ./datasets/Ford/Ford_full/Ford_01_q_1mm/\*.ply

## Eval DPCNet
### Eval DPCNet on SemanticKITTI
chmod +x pc_error



chmod +x tmc3_v29



python encode.py --input_globs ./data/SemanticKITTI/\*.ply --compressed_path ./data/SemanticKITTI/compress/ --datatype semantickitti --gpu_id 0 --K 32 --use_oae



python decode.py  --compressed_path ./data/SemanticKITTI/compress/ --decompressed_path ./data/SemanticKITTI/decompress/ --datatype semantickitti --gpu_id 0 --use_oae



python eval_PSNR.py --input_globs ./data/SemanticKITTI/\*.ply --decompressed_path ./data/SemanticKITTI/decompress/ --datatype semantickitti

### Eval DPCNet on Ford

chmod +x pc_error



chmod +x tmc3_v29



python encode.py --input_globs ./data/Ford/\*.ply --compressed_path ./data/Ford/compress/ --datatype ford --gpu_id 0 --K 64 --use_oae



python decode.py  --compressed_path ./data/Ford/compress/ --decompressed_path ./data/Ford/decompress/ --datatype ford --gpu_id 0 --use_oae



python eval_PSNR.py --input_globs ./data/Ford/\*.ply --decompressed_path ./data/Ford/decompress/ --datatype ford

## Eval LightDPCNet
### Eval LightDPCNet on SemanticKITTI

chmod +x pc_error



chmod +x tmc3_v29



python encode.py --input_globs ./data/SemanticKITTI/\*.ply --compressed_path ./data/SemanticKITTI/compress/ --datatype semantickitti --gpu_id 0 --K 32



python decode.py  --compressed_path ./data/SemanticKITTI/compress/ --decompressed_path ./data/SemanticKITTI/decompress/ --datatype semantickitti --gpu_id 0



python eval_PSNR.py --input_globs ./data/SemanticKITTI/\*.ply --decompressed_path ./data/SemanticKITTI/decompress/ --datatype semantickitti

### Eval LightDPCNet on Ford

chmod +x pc_error



chmod +x tmc3_v29



python encode.py --input_globs ./data/Ford/\*.ply --compressed_path ./data/Ford/compress/ --datatype ford --gpu_id 0 --K 64



python decode.py  --compressed_path ./data/Ford/compress/ --decompressed_path ./data/Ford/decompress/ --datatype ford --gpu_id 0



python eval_PSNR.py --input_globs ./data/Ford/\*.ply --decompressed_path ./data/Ford/decompress/ --datatype ford
