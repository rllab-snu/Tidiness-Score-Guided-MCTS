import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class PybulletNpyDataset(Dataset):
    def __init__(self, data_dir='/home/gun/ssd/disk/ur5_tidying_data/pybullet_line/train', augmentation=False, num_duplication=4):
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.buff_i = None
        self.num_duplication = num_duplication
        self.fsize = 900

        self.find_npydata(self.data_dir)
        #self.current_fidx = 0
        self.load_data() #self.current_fidx)

        # soft labeling
        self.labels = np.linspace(1, 0, self.num_duplication)
    
    def __getitem__(self, index):
        npy_idx = (index // self.fsize) // self.num_file
        current_buff_i = self.buff_i[npy_idx]

        infile_idx = index % self.fsize
        i = current_buff_i[infile_idx]
        i = np.transpose(i, [2, 0, 1])
        i = torch.from_numpy(i).type(torch.float)

        # label
        label_idx = infile_idx % self.num_duplication
        label = torch.from_numpy(self.labels[label_idx:label_idx+1]).type(torch.float)
        return i, label

    def __len__(self):
        return self.num_file * self.fsize

    def find_npydata(self, data_dir):
        rgb_list = [f for f in os.listdir(data_dir) if f.startswith('image_')]
        self.rgb_list = sorted(rgb_list)
        self.num_file = len(self.rgb_list)

    def load_data(self):
        #print('load %d-th npy file.' %dnum)
        buff_i = []
        for rgb_file in self.rgb_list:
            patch_i = np.load(os.path.join(self.data_dir, rgb_file))[:, :, :, :3]
            buff_i.append(patch_i)
        self.buff_i = buff_i

class TabletopTemplateDataset(Dataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', remove_bg=True, label_type='linspace', view='top', scene_index=False, get_mask=False, target_scene=''):
        super().__init__()
        self.data_dir = data_dir
        self.remove_bg = remove_bg
        self.label_type = label_type
        self.scene_index = scene_index
        self.get_mask = get_mask
        self.target_scene = target_scene
        self.view = view
        self.data_paths, self.data_labels = self.get_data_paths()
    
    def get_data_paths(self):
        # B6/template_00001/traj_00092/001/
        #       obj_info.json
        #       rgb_front_top.png
        #       rgb_top.png
        #       depth_front_top.npy
        #       depth_top.npy
        #       seg_front_top.npy
        #       seg_top.npy
        data_paths = []
        data_labels = []
        scene_indices = []
        scene_list = sorted(os.listdir(self.data_dir))
        if self.target_scene!='':
            scene_list = [s for s in scene_list if s.startswith(self.target_scene)]
        for scene in scene_list:
            if scene.startswith('B'):
                scene_index = 0
            elif scene.startswith('C'):
                scene_index = 1
            elif scene.startswith('D'):
                scene_index = 2
            elif scene.startswith('O'):
                scene_index = 3
            else:
                scene_index = -1
            scene_path = os.path.join(self.data_dir, scene)
            for template in sorted(os.listdir(scene_path)):
                template_path = os.path.join(scene_path, template)
                trajectories = sorted(os.listdir(template_path))
                for trajectory in trajectories:
                    trajectory_path = os.path.join(template_path, trajectory)
                    steps = sorted(os.listdir(trajectory_path))
                    num_steps = len(steps)
                    if num_steps!=5:
                        print('skip %s (%d steps)'%(trajectory_path, num_steps))
                        continue
                    if self.label_type == 'linspace':
                        labels = np.linspace(1, 0, num_steps)
                    elif self.label_type == 'binary':
                        labels = [1] + [0] * (num_steps - 1)
                    for i, step in enumerate(steps):
                        data_path = os.path.join(trajectory_path, step)
                        data_paths.append(data_path)
                        data_labels.append(labels[i])
                        scene_indices.append(scene_index)

        if self.scene_index:
            self.scene_indices = scene_indices
        return data_paths, data_labels

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]
        if self.scene_index:
            scene_index = self.scene_indices[index]

        if self.remove_bg:
            rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
            mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
            rgb = rgb * (mask!=mask.max())[:, :, None]
        else:
            rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))

        rgb = np.transpose(rgb[:, :, :3], [2, 0, 1]) / 255.
        rgb = torch.from_numpy(rgb).type(torch.float)
        label = torch.from_numpy(np.array([data_label])).type(torch.float)
        if self.scene_index:
            return rgb, label, scene_index
        else:
            if self.get_mask:
                mask = torch.from_numpy((mask!=mask.max())).type(torch.float)
                return mask, label, rgb
            else:
                return rgb, label

    def __len__(self):
        return len(self.data_paths)


class SimDataset(Dataset):
    def __init__(self, data_dir='/ssd/disk/TableTidyingUp/dataset_template/train', view='top'):
        super().__init__()
        self.data_dir = data_dir
        self.view = view
        self.data_paths = self.get_data_paths()
    
    def get_data_paths(self):
        data_paths = []
        for scene in sorted(os.listdir(self.data_dir)):
            scene_path = os.path.join(self.data_dir, scene)
            for template in sorted(os.listdir(scene_path)):
                template_path = os.path.join(scene_path, template)
                trajectories = sorted(os.listdir(template_path))
                for trajectory in trajectories:
                    trajectory_path = os.path.join(template_path, trajectory)
                    steps = sorted(os.listdir(trajectory_path))
                    num_steps = len(steps)
                    for step in steps[:min(3, num_steps)]:
                        data_path = os.path.join(trajectory_path, step)
                        data_paths.append(data_path)
        return data_paths

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        rgb = np.array(Image.open(os.path.join(data_path, 'rgb_%s.png'%self.view)))
        mask = np.load(os.path.join(data_path, 'seg_%s.npy'%self.view))
        #rgb = rgb * (mask!=mask.max())[:, :, None]
        return rgb, mask

    def __len__(self):
        return len(self.data_paths)
