import numpy as np
import torch
from mask_4d.utils.voxelize import voxelize
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min, scatter_sum

def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc


def convert_to1(data):
    mean = np.mean(data, axis=0)
    data -= mean
    m = np.max(np.abs(data))
    data /= m
    return data


def shuffle_points(pc):
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    return pc[idx,:]


def rotate_point_cloud(pc):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (1, C))
    pc += shifts
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # Set to the first point
    return pc


def augment_pc(pc):
    rotated_pc = rotate_point_cloud(pc[:, :3])
    jittered_pc = shift_point_cloud(rotated_pc)
    jittered_pc = jitter_point_cloud(jittered_pc)
    pc[:, :3] = jittered_pc
    return pc

import hashlib

def string_to_number(track_id):
    hash_object = hashlib.sha256(track_id.encode())
    hex_dig = hash_object.hexdigest()
    return int(hex_dig, 16)

def count_digits(number):
    return len(str(number))

def safe_convert_track_id(chosen_data):
    track_ids = chosen_data[:]['track_id']
    # Replace empty string with a specific default value, such as 'unknown'
    track_ids = np.where(track_ids == b'', -1, track_ids)
    # Convert string to number
    ins_labels = np.array([string_to_number(id.decode()) for id in track_ids], dtype=np.int64)
    return ins_labels

def hash_string_to_int(s, max_value=65535):
    """Generate integer index using hash function"""
    # Convert bytes type to string
    if isinstance(s, np.bytes_):
        s = s.decode()
    hash_value = int(hashlib.md5(s.encode()).hexdigest(), 16)
    return hash_value % max_value

def replace_empty_with_special(track_ids, special_char=" ", max_value=1000000):
    # Generate hash index for special character
    special_index = hash_string_to_int(special_char, max_value)
    
    # Ensure track_ids are of string type
    if isinstance(track_ids, np.ndarray) and track_ids.dtype.type is np.bytes_:
        track_ids = [id.decode() for id in track_ids]
    
    # Replace empty string with special character
    track_ids_int = [hash_string_to_int(id, max_value) & 0xFFFF for id in track_ids]  # Use & 0xFFFF to ensure within 0 to 65535
    return track_ids_int


def data_prepare(
    coord,
    feat,
    label,
    split="train",
    voxel_size=np.array([0.1, 0.1, 0.1]),
    voxel_max=None,
):
    coord_min = np.min(coord, 0)
    coord_norm = coord - coord_min
    if split == "train":
        uniq_idx, idx_recon = voxelize(coord_norm, voxel_size)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(voxel_size))
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
        if voxel_max and label.shape[0] > voxel_max:
            init_idx = np.random.randint(label.shape[0])
            crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[
                :voxel_max
            ]
            coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
            coord_voxel = coord_voxel[crop_idx]
    else:
        idx_recon = voxelize(coord_norm, voxel_size, mode=1)

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    if split == "train":
        coord_voxel = torch.LongTensor(coord_voxel)
        idx_recon = torch.LongTensor(idx_recon)
        return coord_voxel, coord, feat, label, idx_recon
    else:
        coord_norm = torch.FloatTensor(coord_norm)
        idx_recon = torch.LongTensor(idx_recon)
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coord_voxel = torch.floor(coord_norm / torch.from_numpy(voxel_size)).long()
        coord = scatter_mean(coord, idx_recon, dim=0)
        feat = scatter_mean(feat, idx_recon, dim=0)
        return coord_voxel, coord, feat, label, idx_recon

if __name__ == '__main__':
    # Example
    track_id = '2ed727c6c2e311e6a58b485b3918780b'
    track_ids = np.array([b'', b'13ed213caa8511e89373ecf4bb1ae69c', b'13ed213caa8511e89373ecf4bb1ae69c', b'2ed727c6c2e311e6a58b485b3918780b', b''], dtype=np.bytes_)
    processed_ids = replace_empty_with_special(track_ids)

    print(processed_ids)
