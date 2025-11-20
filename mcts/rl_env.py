import os
import sys
import copy
import random
import json
import numpy as np
import pybullet as p
from argparse import ArgumentParser
from matplotlib import pyplot as plt

import torch
from data_loader import SimDataset
from utils import loadRewardFunction, Renderer

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import TableTopTidyingUpEnv, get_contact_objects
from utilities import Camera, Camera_front_top

        
class Environment(object):
    def __init__(self, args):
        self.renderer = Renderer(
                                tableSize=(args.H, args.W), 
                                imageSize=(360, 480), 
                                cropSize=(args.crop_size, args.crop_size)
                                )
        self.setVNet(args.reward_model_path)
        self.data_dir = args.data_dir
        self.gui_on = not args.gui_off
        self.num_objects = args.num_objects
        self.batchSize = args.batch_size
        self.thresholdSuccess = args.threshold_success
        self.maxLength = args.max_length
        
        self.use_template = args.use_template
        self.sim = args.sim
        self.datasetDir = args.dataset_dir
        self.real = args.real
        
        if self.sim:
            self.dataset = SimDataset(data_dir=self.datasetDir, view='top')
        else:
            self.objects = self.setObjects()
            self.tableEnv = self.setupTableEnvironment()

        self.state = None
        self.rewardType = args.reward_type
        self.reward = 0.0

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def reset(self):
        self.countStep = 0
        if self.sim:
            while True:
                sceneIdx = np.random.choice(len(self.dataset))
                rgb, seg = self.dataset[sceneIdx]
                # objects: 2~N+1 -> 4~N+3
                # table: N+2->1
                seg = seg + 2
                seg[seg==seg.max()] = 1
                initRgb, initSeg = rgb, seg
                # Check occlusions
                is_occluded = self.checkOcclusion(initSeg)
                if is_occluded: continue
                else: break
        else:
            self.tableEnv.reset()
            if self.use_template:
                if 'unseen' in [args.scene_split, args.object_split]:
                    dataset = f'test-{args.object_split}_obj-{args.scene_split}_template'
                else: 
                    dataset = 'train'
                template_folder = os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/templates')
                template_files = os.listdir(template_folder)
                template_files = [f for f in template_files if f.lower().endswith('.json')]
                
                template_file = random.choice(template_files)
                # scene = template_file.split('_')[0]
                # template_id = template_file.split('_')[-1].split('.')[0]
                with open(os.path.join(template_folder, template_file), 'r') as f:
                    templates = json.load(f)
                augmented_template = env.get_augmented_templates(templates, 2)[-1]
                objects = [v for k,v in augmented_template['objects'].items()]
                if len(objects)<self.num_objects:
                    selected_objects = [objects[i] for i in np.random.choice(len(objects), self.num_objects, replace=True)]
                else:
                    selected_objects = [objects[i] for i in np.random.choice(len(objects), self.num_objects, replace=False)]
                # env.load_template(augmented_template)
            else:
                selected_objects = [self.objects[i] for i in np.random.choice(len(self.objects), self.num_objects, replace=False)]
            
            self.tableEnv.spawn_objects(selected_objects)
            while True:
                self.tableEnv.arrange_objects(random=True)
                obs = self.tableEnv.get_observation()
                initRgb = obs['top']['rgb']
                initSeg = obs['top']['segmentation']
                # Check occlusions
                is_occluded = self.checkOcclusion(initSeg)
                # Check collision
                is_collision = self.checkCollision()
                if is_occluded or is_collision: continue
                else: break
        table = self.renderer.setup(initRgb, initSeg)
        self.previousTUscore = self.getReward([table])[0]
        self.currentTable = table

        obs = self.getObservation(table)
        return obs
        # currentRgb = self.renderer.getRGB(table)
        # return currentRgb/255., table

    def getObservation(self, table):
        rgb = self.renderer.getRGB(table)/255.
        rgbWoTargets = []
        objectPatches = []
        
        for o in range(self.renderer.numObjects):
            rgbWoTarget = self.renderer.getRGB(table, remove=o)/255.
            rgbWoTargets.append(rgbWoTarget)
        for r in range(1,3):
            for o in range(self.renderer.numObjects):
                objPatch = self.renderer.objectPatches[r][o]/255.
                objectPatches.append(objPatch)
        return rgb, rgbWoTargets, objectPatches

    def step(self, action):
        self.countStep += 1
        # action: (object, y, x, rotation)
        if self.real:
            self.previousTable = copy.deepcopy(self.currentTable)
            target_object, target_position, rot_angle = self.renderer.convert_action(action)
            tableobs = self.tableEnv.step(target_object, target_position, rot_angle)
            newRgb = tableobs['top']['rgb']
            newSeg = tableobs['top']['segmentation']
            newTable = self.renderer.setup(newRgb, newSeg)
            if newTable is None or self.renderer.numObjects!=self.num_objects:
                reward = -1.0
                success = False
                terminal = True
                empty_rgb = np.zeros([self.num_objects, 360, 480, 3])
                empty_patch = np.zeros([2*self.num_objects, 128, 128, 3])
                obs = [newRgb, empty_rgb, empty_patch]
                return obs, reward, success, terminal
        else:
            self.previousTable = copy.deepcopy(self.currentTable)
            obj, py, px, rot = action
            posMap, rotMap = self.previousTable
            newPosMap = copy.deepcopy(posMap)
            newRotMap = copy.deepcopy(rotMap)
            newPosMap[posMap==obj] = 0
            newPosMap[py, px] = obj
            newRotMap[posMap==obj] = 0
            newRotMap[py, px] = rot
            newTable = [newPosMap, newRotMap]
        reward, success, terminal = self.isTerminal(newTable)
        self.currentTable = newTable
        
        obs = self.getObservation(newTable)
        return obs, reward, success, terminal
        # currentRgb = self.renderer.getRGB(newTable)
        # return [currentRgb/255., newTable], reward, success, terminal

    def isTerminal(self, table):
        # print('isTerminal')
        success = False
        terminal = False
        reward = 0.0
        # check collision and reward
        collision = self.renderer.checkCollision(table)
        if collision:
            reward = -1.0
            terminal = True
        else:
            TUscore = self.getReward([table])[0]
            if self.rewardType.startswith('delta'):
                reward = TUscore - self.previousTUscore
            elif self.rewardType=='binary':
                if TUscore > self.previousTUscore: reward = 0.1
                else: reward = -0.1
            if TUscore > self.thresholdSuccess:
                reward = 1.0
                success = True
                terminal = True
            self.previousTUscore = TUscore
        if self.countStep >= self.maxLength:
            terminal = True
        return reward, success, terminal
    
    def getReward(self, tables):
        # print('getReward.')
        states = []
        for table in tables:
            rgb = self.renderer.getRGB(table)
            states.append(rgb)
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)

        if len(states) > self.batchSize:
            rewards = []
            numBatches = len(states)//self.batchSize
            if len(states)%self.batchSize > 0:
                numBatches += 1
            for b in range(numBatches):
                batchS = s[b*self.batchSize:(b+1)*self.batchSize]
                batchRewards = self.VNet(batchS).cpu().detach().numpy()
                rewards.append(batchRewards)
            rewards = np.concatenate(rewards)
        else:
            rewards = self.VNet(s).cpu().detach().numpy()
        return rewards.reshape(-1)

    def checkOcclusion(self, segmentation):
        for o in range(self.num_objects):
            # get the segmentation mask of each object #
            mask = (segmentation==o+4).astype(float)
            if mask.sum()==0: return True
        return False
    
    def checkCollision(self):
        contact_objects = get_contact_objects()
        contact_objects = [c for c in list(get_contact_objects()) if 1 not in c and 2 not in c]
        if len(contact_objects) > 0: return True
        else: return False

    def setVNet(self, model_path):
        vNet, preprocess = loadRewardFunction(model_path)
        self.VNet = vNet
        self.preProcess = preprocess

    def setObjects(self):
        objects = ['book', 'bowl', 'can_drink', 'can_food', 'cleanser', 'cup', 'fork', 'fruit', 'glass', \
                            'glue', 'knife', 'lotion', 'marker', 'pitcher', 'plate', 'remote', 'scissors', 'shampoo', \
                            'soap', 'soap_dish', 'spoon', 'stapler', 'teapot', 'timer', 'toothpaste']
        # objects = ['bowl', 'can_drink', 'plate', 'marker', 'soap_dish', 'book', 'remote', 'fork', 'knife', 'spoon', 'teapot', 'cup']
        objects = [(o, 'medium') for o in objects]
        return objects

    def setupTableEnvironment(self):
        #selected_objects = [self.objects[i] for i in np.random.choice(len(self.objects), self.num_objects, replace=False)]
        camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
        camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
        
        objects_cfg = { 'paths': {
                'pybullet_object_path' : os.path.join(self.data_dir, 'pybullet-URDF-models/urdf_models/models'),
                'ycb_object_path' : os.path.join(self.data_dir, 'YCB_dataset'),
                'housecat_object_path' : os.path.join(self.data_dir, 'housecat6d/obj_models_small_size_final'),
            },
            'split' : 'inference' #'train'
        }
        tableEnv = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=self.gui_on, num_objs=self.num_objects, gripper_type='85')
        p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
        p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

        tableEnv.reset()
        # tableEnv.spawn_objects(selected_objects)
        # tableEnv.arrange_objects(random=True)
        return tableEnv


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=128)
    parser.add_argument('--real', action="store_true")
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--sim', action="store_true")
    parser.add_argument('--dataset_dir', type=str, default='/ssd/disk/TableTidyingUp/dataset_template/train')
    parser.add_argument('--max-length', type=int, default=20)
    parser.add_argument('--threshold-success', type=float, default=0.9)
    parser.add_argument('--reward-model-path', type=str, default='data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--reward-type', type=str, default='delta-reward') # 'delta-rewrad' / 'binary'
    parser.add_argument('--batch-size', type=int, default=32)
    
    parser.add_argument('--num-objects', type=int, default=4)
    parser.add_argument('--num-scenes', type=int, default=10)
    args = parser.parse_args()

    env = Environment(args)
    for s in range(args.num_scenes):
        obs = env.reset()
        for i in range(args.max_length):
            print('step:', i)
            n = np.random.choice(np.arange(1, args.num_objects+1))
            y = np.random.choice(args.H)
            x = np.random.choice(args.W)
            rot = np.random.choice([1, 2])
            action = (n, y, x, rot)
            print('action:', action)

            obs, reward, success, terminal = env.step(action)
            rgb, rgbWoTargets, objPatches = obs
            
            # print('rgb:', obs)
            plt.imshow(obs[0]/255.)
            plt.imshow(obs[1][0]/255.)
            plt.imshow(obs[2][0]/255.)
            plt.show()
            if terminal:
                print('terminal:', terminal)
                print('Episode finished.')
                break
    
