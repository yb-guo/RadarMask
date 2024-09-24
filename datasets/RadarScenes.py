import h5py
import json
import numpy as np
import os
import sys
import math
import warnings
import bisect
import time
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.provider import augment_pc, pc_normalize, random_point_dropout, convert_to1, replace_empty_with_special, data_prepare
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch

class RadarScenes():
    def __init__(self, data_root, split, npoints, combined_frame_num, augment=True, dp=False, normalize=False, voxel_size=0.01, voxel_max=1):
        assert(split == 'train' or split == 'validation')
        self.split = split
        self.npoints = npoints
        self.combined_frame_num = combined_frame_num
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        self.voxel_size = np.array(voxel_size)
        self.voxel_max = voxel_max
        self.th_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.min_points = 0

        start_time = time.time()  # Start timing

        with open(os.path.join(data_root, 'sequences.json'), 'r') as f:
            sequences = json.load(f)
            # generate train and validation lists
            train_radar_data_lists = []
            validation_radar_data_lists = []
            train_scenes_lists = []
            validation_scenes_lists = []
            for index in range(len(sequences['sequences'])):
                if sequences['sequences']['sequence_'+str(index+1)]['category'] == 'train':
                    # append path to train list
                    train_radar_data_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'radar_data.h5'))
                    train_scenes_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'scenes.json'))
                elif sequences['sequences']['sequence_'+str(index+1)]['category'] == 'validation':
                    # append path to validation list
                    validation_radar_data_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'radar_data.h5'))
                    validation_scenes_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'scenes.json'))

        self.h5_file_lists = []
        self.json_file_lists = []
        if split == 'train':
            self.h5_file_lists.extend(train_radar_data_lists)
            self.json_file_lists.extend(train_scenes_lists)
            # reduce the number of files for debug
            self.h5_file_lists = self.h5_file_lists[:1]
            self.json_file_lists = self.json_file_lists[:1]
        elif split == 'validation':
            self.h5_file_lists.extend(validation_radar_data_lists)
            self.json_file_lists.extend(validation_scenes_lists)
            # reduce the number of files for debug
            self.h5_file_lists = self.h5_file_lists[:1]
            self.json_file_lists = self.json_file_lists[:1]

        self.file_num = len(self.h5_file_lists)
        # number of combined frames in all files
        self.file_combined_frame_num_list = self.get_file_combined_frame_num_list()
        self.seg_classes = {'Scenes': list(range(12))}  # 11 classes
        self.caches = {}

        end_time = time.time()  # End timing

        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        # Add time measurement here
        start_time = time.time()  # Start timing
        self.timestamp_to_odometry = []
        self.map_radar_to_odometry()
        end_time = time.time()  # End timing

        elapsed_time = end_time - start_time
        print(f"Time taken to map radar to odometry: {elapsed_time:.2f} seconds")

    def find_nearest_timestamp(self, odom_timestamps, target_timestamp):
        pos = bisect.bisect_left(odom_timestamps, target_timestamp)
        if pos == 0:
            return 0
        if pos == len(odom_timestamps):
            return len(odom_timestamps) - 1
        before = odom_timestamps[pos - 1]
        after = odom_timestamps[pos]
        if after - target_timestamp < target_timestamp - before:
            return pos
        else:
            return pos - 1

    def map_radar_to_odometry(self):
        """Pre-calculate the closest odometry data for each radar timestamp and record the time offset."""
        time_differences = []  # Store the time offset for each timestamp

        for file_index in range(self.file_num):
            h5_file = h5py.File(self.h5_file_lists[file_index], 'r')
            radar_data = h5_file['radar_data'][:]
            odometry_data = h5_file['odometry'][:]

            radar_timestamps = radar_data['timestamp']
            odom_timestamps = odometry_data['timestamp']

            nearest_odom_indices = [self.find_nearest_timestamp(odom_timestamps, ts) for ts in radar_timestamps]
            chosen_odometry = odometry_data[nearest_odom_indices]
            
            self.timestamp_to_odometry.append(chosen_odometry)

            # Calculate the time offset
            for radar_ts, odom_idx in zip(radar_timestamps, nearest_odom_indices):
                odom_ts = odom_timestamps[odom_idx]
                time_diff = abs(radar_ts - odom_ts)
                time_differences.append(time_diff)

        # Display time offset statistics
        max_diff = np.max(time_differences)
        min_diff = np.min(time_differences)
        mean_diff = np.mean(time_differences)
        std_diff = np.std(time_differences)
        max_diff_sec = max_diff / 1_000_000
        min_diff_sec = min_diff / 1_000_000
        mean_diff_sec = mean_diff / 1_000_000
        std_diff_sec = std_diff / 1_000_000
        print(f"Time offset statistics:")
        print(f"Max time offset: {max_diff_sec:.6f} seconds")
        print(f"Min time offset: {min_diff_sec:.6f} seconds")
        print(f"Average time offset: {mean_diff_sec:.6f} seconds")
        print(f"Time offset standard deviation: {std_diff_sec:.6f} seconds")
    
    # Other functions...

    def __getitem__(self, index):
        file_count = 0
        frame_index = None

        # Find the file index and frame index
        for file_index, num in enumerate(self.file_combined_frame_num_list):
            if index <= file_count + (num - 1):
                frame_index = index - file_count
                break
            else:
                file_count += num

        if frame_index is not None:
            h5_file = h5py.File(self.h5_file_lists[file_index], 'r')
            radar_data = h5_file['radar_data'][:]

            json_file = json.load(open(self.json_file_lists[file_index], 'r'))
            fname = json_file.get('sequence_name', 'Unknown')
            frame_list = list(json_file['scenes'].keys())
            # Get selected frames
            chosen_frames = frame_list[frame_index * self.combined_frame_num:(frame_index + 1) * self.combined_frame_num]
            if len(chosen_frames) < self.combined_frame_num:
                chosen_frames = frame_list[-self.combined_frame_num:]
            chosen_frames = [int(ts) for ts in chosen_frames]
            chosen_data = radar_data[np.isin(radar_data['timestamp'], chosen_frames)]

            # Extract radar timestamps
            radar_timestamps = chosen_data['timestamp']

            # Extract odometry data corresponding to radar timestamps
            radar_timestamps = np.unique(chosen_data['timestamp'])
            radar_indices = [np.where(radar_timestamps == ts)[0][0] for ts in radar_timestamps]
            chosen_odometry = self.timestamp_to_odometry[file_index][radar_indices]

            full_xyz = np.zeros((len(chosen_data), 3)).astype(np.float32)
            full_xyz[:, 0] = chosen_data[:]['x_cc']
            full_xyz[:, 1] = chosen_data[:]['y_cc']
            full_xyz[:, 2] = 0  # keep z as 0
            xyz_points = np.zeros((len(chosen_data), 11)).astype(np.float32)
            xyz_points[:, 0] = chosen_data[:]['x_cc']
            xyz_points[:, 1] = chosen_data[:]['y_cc']
            xyz_points[:, 2] = 0  # keep z as 0

            # Coordinates
            xyz_points[:, 3] = chosen_data[:]['vr_compensated']
            xyz_points[:, 4] = chosen_data[:]['rcs']
            # Features
            xyz_points[:, 5] = chosen_odometry['x_seq']
            xyz_points[:, 6] = chosen_odometry['y_seq']
            xyz_points[:, 7] = chosen_odometry['yaw_seq']
            xyz_points[:, 8] = chosen_odometry['vx']
            xyz_points[:, 9] = chosen_odometry['yaw_rate']
            # Odometry information
            xyz_points[:, 10] = chosen_odometry['timestamp']
            sem_labels = chosen_data[:]['label_id'].astype(np.int32)
            ins_labels = replace_empty_with_special(chosen_data[:]['track_id'])
            odometry = np.zeros((len(choice), 6)).astype(np.float32)
            odometry[:, 0] = xyz_points[:, 5]  # x_seq
            odometry[:, 1] = xyz_points[:, 6]  # y_seq
            odometry[:, 2] = xyz_points[:, 7]  # yaw_seq
            odometry[:, 3] = xyz_points[:, 8]  # vx
            odometry[:, 4] = xyz_points[:, 9]  # yaw_rate
            odometry[:, 5] = xyz_points[:, 10]  # timestamp

            sem_labels = self.label_translation(sem_labels)  # Translate the semantic labels

            return xyz, sem_labels, ins_labels, xyz_points, feats, odometry


    def plot_combined_frame(self, index):
        import matplotlib.pyplot as plt
        xyz_points, labels = self.__getitem__(index)

        plt.scatter(xyz_points[:, 1], xyz_points[:, 0], c=labels, s=5)
        # reverse the x axis
        plt.xlim(plt.xlim()[::-1])
        plt.show()

class RadarScenesDataset(Dataset):
    def __init__(self, radar_scenes, split='train'):
        self.radar_scenes = radar_scenes
        self.split = split

    def __len__(self):
        return len(self.radar_scenes)

    def __getitem__(self, index):
        xyz,sem_labels,ins_labels,masks,masks_cls,masks_ids,fname,odometry,full_xyz,full_xyz,sp_coords,sp_xyz,sp_feats,sp_labels,sp_idx_recons = self.radar_scenes[index]


        # return {
        #     "pt_coord":xyz,
        #     "sem_label":sem_labels,
        #     "ins_label":ins_labels,
        #     "masks":masks,
        #     "masks_cls":masks_cls,
        #     "masks_ids":masks_ids,
        #     "fname":fname,
        #     "pose":odometry,
        #     "keep_xyz":full_xyz,
        #     "full_xyz":full_xyz,
        #     "sp_coord":sp_coords,
        #     "sp_xyz":sp_xyz,
        #     "sp_feat":sp_feats,
        #     "sp_label":sp_labels,
        #     "sp_idx_recons":sp_idx_recons
        # }
        return (
            xyz,sem_labels,ins_labels,masks,masks_cls,masks_ids,fname,odometry,full_xyz,full_xyz,sp_coords,sp_xyz,sp_feats,sp_labels,sp_idx_recons
        )
    
class RadarScenesDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=32, num_workers=4, npoints=256, combined_frame_num=1, augment=True, dp=False, normalize=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.npoints = npoints
        self.combined_frame_num = combined_frame_num
        self.augment = augment
        self.dp = dp
        self.normalize = normalize
        # self.collate_fn = SphericalSequenceCollation()


    def setup(self, stage=None):
        # 训练集
        self.train_data = RadarScenes(data_root=self.data_root, split='train', npoints=self.npoints, combined_frame_num=self.combined_frame_num, augment=self.augment, dp=self.dp, normalize=self.normalize)
        self.train_dataset = RadarScenesDataset(self.train_data, split='train')
        
        # 验证集
        self.val_data = RadarScenes(data_root=self.data_root, split='validation', npoints=self.npoints, combined_frame_num=self.combined_frame_num, augment=False, dp=False, normalize=False)
        self.val_dataset = RadarScenesDataset(self.val_data, split='validation')
        
        # 测试集
        # self.test_data = RadarScenes(data_root=self.data_root, split='test', npoints=self.npoints, combined_frame_num=self.combined_frame_num, augment=False, dp=False, normalize=False)
        # self.test_dataset = RadarScenesDataset(self.test_data, split='test')

    def train_dataloader(self):
         return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn = SphericalSequenceCollation(), shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn = SphericalSequenceCollation(),shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # 如果有测试集
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        pass


if __name__ == '__main__':
    radarscenes_data_module = RadarScenesDataModule(data_root='/home/root123/0_datasets/RadarScenes/data', batch_size=32, npoints=256, combined_frame_num=1, augment=True)
    radarscenes_data_module.setup()
    train_loader = radarscenes_data_module.train_dataloader()
    cnt = 0 
    for batch in train_loader:
        cnt = cnt+1
        # print(cnt)
