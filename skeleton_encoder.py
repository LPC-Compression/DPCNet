import torch
import numpy as np

import sys
sys.path.append("./NumpyAc")
import numpyAc

from torch.utils.data import DataLoader
import torch.nn.functional as F
import subprocess

best_octformer_model = torch.load('./models/kitti_best.pth').cuda()
best_octformer_model.eval()
best_octformer_model.to(torch.float64)

# 节点，构成OCtree的基本元素
class Octantids:
    def __init__(self, children, center, extent, depth:int, is_leaf:bool, curIndex:int, parentOccupancy:int):
        self.children = children  # 子节点
        self.center = center  # 正方体的中心点
        self.extent = extent  # 正方体的边长一半
        self.is_leaf = is_leaf  # 是否叶节点
        self.depth = depth  # 节点的深度
        self.octant = 0  # octant，代表八个子节点内是否有点，0-255（即00000000-11111111）
        self.curIndex = curIndex
        self.parentOccupancy = parentOccupancy

def flattenFeatures(node):
    return (node.center[0],node.center[1],node.center[2], node.depth,node.curIndex, node.parentOccupancy)

def octree_BFS_build(db_np:np.ndarray,depth:int):
    if len(db_np)==0:
       return None,np.empty([0,3])     
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = (db_np_max+db_np_min)/2
    db_depth = 1
    point_indices=list(range(N))
    root = Octantids([None for i in range(8)], db_center, db_extent, db_depth, is_leaf=False, curIndex=0, parentOccupancy=0)
    pointbuildinginfo=[point_indices]
    leafnodeList=[root]
    def octree_split_nextlayer():
        nonlocal leafnodeList,pointbuildinginfo,db_np
        point_indices=pointbuildinginfo.pop(0)
        node = leafnodeList.pop(0)
        center=node.center
        extent=node.extent
        depth=node.depth
        node.is_leaf=False
        children_point_indices = [[] for i in range(8)]
        for point_idx in point_indices:
            point_db = db_np[point_idx]
            morton_code = 0
            if point_db[0] > center[0]:
                morton_code = morton_code | 1
            if point_db[1] > center[1]:
                morton_code = morton_code | 2
            if point_db[2] > center[2]:
                morton_code = morton_code | 4
            children_point_indices[morton_code].append(point_idx)
        for i in range(8):
            if (len(children_point_indices[i]) > 0):
                node.octant += 2 ** (7 - i)
        factor = [-0.5, 0.5]
        for i in range(8):
            if len(children_point_indices[i])>0:
                child_center_x = center[0] + factor[(i & 1) > 0] * extent
                child_center_y = center[1] + factor[(i & 2) > 0] * extent
                child_center_z = center[2] + factor[(i & 4) > 0] * extent
                child_extent = 0.5 * extent
                child_center = np.asarray([child_center_x, child_center_y, child_center_z], dtype=np.float64)
                node.children[i] = Octantids([None for i in range(8)], child_center, child_extent, depth+1, is_leaf=True, curIndex=i, parentOccupancy=node.octant)            
                pointbuildinginfo.append(children_point_indices[i])
                leafnodeList.append(node.children[i]) 
    current_layer=1

    node_count = [1]
    data = []
    label = []

    while current_layer<depth:
        NodeNum=len(leafnodeList)   

        data.append(np.array([flattenFeatures(node) for node in leafnodeList], dtype=np.float64))
        nodeList_copy = [item for item in leafnodeList]

        for _ in range(NodeNum):
            octree_split_nextlayer()
        current_layer+=1

        label.append(np.array([node.octant for node in nodeList_copy], dtype=np.float64))   
        node_count.append(len(leafnodeList))

    rec_pc = np.array([node.center for node in leafnodeList], dtype=np.float32)
    return root, db_extent, db_center, np.array(node_count), rec_pc, data, label

def encode(pdf, sym, binpath):
    codec = numpyAc.arithmeticCoding()
    byte_stream, real_bits = codec.encode(pdf, sym, binpath)
    return byte_stream, real_bits

def octformer_encode_backbone(pc, save_path, depth=12, seq_size=512, batch_size=64):
    min_, max_ = np.amin(pc), np.amax(pc)
    norm_point_cloud = (pc - min_) / (max_ - min_)
    root, db_extent, db_center, node_count, rec_pc, data, label = octree_BFS_build(db_np=norm_point_cloud, depth=depth)
    data = [item.astype(np.float32) for item in data]

    node_count = node_count[0:-1]
    roundoff_node_count = (np.ceil(node_count/seq_size) * seq_size).astype(np.int32)
    start = np.cumsum(np.concatenate([np.array([0]), roundoff_node_count[0:-1]]))
    end = start + node_count

    padded_data = [item if len(item)%seq_size==0 else \
    np.concatenate([item, np.zeros(shape=(seq_size - (len(item)%seq_size) ,6), dtype=np.float32)]) for item in data]
    padded_data = np.concatenate(padded_data, axis=0).reshape(-1, seq_size, 6).astype(np.float64)
    dataloader = DataLoader(dataset=padded_data, batch_size=batch_size, drop_last=False, shuffle=False)

    pdflist = []
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.cuda()
            pred = best_octformer_model(batch_data) # B, S, D
            pred = F.softmax(pred, dim=-1).reshape(-1, pred.shape[-1]).detach().cpu().numpy()
            pdflist.append(pred)
    pdflist=np.concatenate(pdflist, axis=0).astype(np.float32)
    pdflist = [pdflist[s:t] for s,t in zip(start, end)]
    total_bits = 0
    for i in range(1, depth-1):
        _, bits = encode(pdflist[i], label[i].astype(np.int16), binpath=save_path + f".{i}.b")
        total_bits += bits
    rec_pc = rec_pc *  (max_ - min_) + min_
    return total_bits, rec_pc, root.octant, min_, max_, db_center, db_extent

def construct_next_layer(node_queue:list):
    N = len(node_queue)
    for _ in range(N):
        node = node_queue.pop(0)
        depth = node.depth
        octant = node.octant
        center=node.center
        extent=node.extent
        factor = [-0.5, 0.5]
        for i in range(8):
            if octant >= 2 ** (7 - i):
                octant -= 2 ** (7 - i)
                child_center_x = center[0] + factor[(i & 1) > 0] * extent
                child_center_y = center[1] + factor[(i & 2) > 0] * extent
                child_center_z = center[2] + factor[(i & 4) > 0] * extent
                child_extent = 0.5 * extent
                child_center = np.asarray([child_center_x, child_center_y, child_center_z], dtype=np.float64)
                new_node = Octantids([None for i in range(8)], child_center, child_extent, depth+1, is_leaf=True, curIndex=i, parentOccupancy=node.octant)
                node_queue.append(new_node)
    return node_queue

def decode(pdf, binpath):
    symsNum, dim = pdf.shape
    decodec = numpyAc.arithmeticDeCoding(None, symsNum, dim, binpath)
    result = []
    for i in range(symsNum):
        result.append(decodec.decode(pdf[i:i+1, :]))
    return np.array(result)


def octformer_decode_backbone(save_path, rec_octant, rec_db_center0, rec_db_center1, rec_db_center2, \
                              rec_db_extent, min_, max_,
                              depth=12, seq_size=512, batch_size=64):
    rec_root = Octantids([None for i in range(8)], np.array([rec_db_center0, rec_db_center1, rec_db_center2]), \
                     rec_db_extent, 1, is_leaf=False, curIndex=0, parentOccupancy=0)
    rec_root.octant = rec_octant
    node_queue = [rec_root]
    for i in range(1, depth-1):
        node_queue = construct_next_layer(node_queue)
        rec_data = np.array([flattenFeatures(node) for node in node_queue], dtype=np.float32)
        rec_data = rec_data.astype(np.float32)
        len_rec_data = rec_data.shape[0]
        padded_rec_data = rec_data if len(rec_data)%seq_size==0 else \
                np.concatenate([rec_data, np.zeros(shape=(seq_size - (len(rec_data)%seq_size) ,6), dtype=np.float32)])
        padded_rec_data = padded_rec_data.reshape(-1, seq_size, 6).astype(np.float64)
        dataloader = DataLoader(dataset=padded_rec_data, batch_size=batch_size, drop_last=False, shuffle=False)
        result_pdf = []
        with torch.no_grad():
            for _, batch_data in enumerate(dataloader):
                batch_data = batch_data.cuda()
                pred = best_octformer_model(batch_data) # B, S, D
                pred = F.softmax(pred, dim=-1).reshape(-1, pred.shape[-1]).detach().cpu().numpy()
                result_pdf.append(pred)
        result_pdf=np.concatenate(result_pdf, axis=0)[0:len_rec_data].astype(np.float32)
        decode_sym = decode(result_pdf, save_path + f".{i}.b")
        for j in range(len_rec_data):
            node_queue[j].octant = decode_sym[j]
    node_queue = construct_next_layer(node_queue)
    decode_pc = np.array([node.center for node in node_queue], dtype=np.float32)
    decode_pc = decode_pc * (max_ - min_)  + min_
    return decode_pc



def tmc_compress(tmc_path, input_file, output_file, scale=1):
    """
        Compress point cloud losslessly using MPEG G-PCCv23 PredTree. 
    """
    cmd = (tmc_path+ 
            ' --mode=0' + 
            ' --geomTreeType=0' + # 0 for octree, 1 for predtree
            ' --mergeDuplicatedPoints=1' +
            f' --positionQuantizationScale=1' +
            f' --uncompressedDataPath='+input_file +
            f' --compressedStreamPath='+output_file)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    slice_number = int(str(output).split('Slice number:')[1].split('\\n')[0])

    xyz_steam_size, xyz_time = 0, 0
    for i in range(slice_number):
        xyz_steam_size += float(str(output).split('positions bitstream size')[i+1].split('B')[0]) * 8
        xyz_time += float(str(output).split('positions processing time (user):')[i+1].split('s')[0])
    return xyz_steam_size, xyz_time

def tmc_decompress(tmc_path, input_file, output_file):
    """
        Decompress point cloud using MPEG G-PCCv23. 
    """
    cmd = (tmc_path+ 
        ' --mode=1'+ 
        ' --compressedStreamPath='+input_file+ 
        ' --reconstructedDataPath='+output_file)
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    xyz_time = float(str(output).split('positions processing time (user):')[1].split('s')[0])
    return xyz_time

