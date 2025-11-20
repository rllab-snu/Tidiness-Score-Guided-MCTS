import os
import sys
import copy
import cv2
import datetime
import time
import random
import numpy as np
import logging
import json
import pybullet as p
from argparse import ArgumentParser
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import suppress_stdout

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_nvisii_env import TableTopTidyingUpEnv, get_contact_objects
from utilities import Camera, Camera_front_top
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp'))
from collect_template_list import scene_list

import wandb
import warnings
warnings.filterwarnings("ignore")


def setupEnvironment(args):
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    data_dir = args.data_dir
    objects_cfg = { 'paths': {
            'pybullet_object_path' : os.path.join(data_dir, 'pybullet-URDF-models/urdf_models/models'),
            'ycb_object_path' : os.path.join(data_dir, 'YCB_dataset'),
            'housecat_object_path' : os.path.join(data_dir, 'housecat6d/obj_models_small_size_final'),
        },
        'split' : args.object_split #'inference' #'train'
    }
    
    gui_on = not args.gui_off
    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=gui_on, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    #env.set_floor(texture_id=-1)
    env.reset()
    return env


if __name__=='__main__':
    parser = ArgumentParser()
    # Data directory
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    # Inference
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--use-template', action="store_true")
    parser.add_argument('--scenes', type=str, default='') # e.g., 'B2,B5,C4,C6,C12,D5,D8,D11,O3,O7'
    parser.add_argument('--inorder', action="store_true")
    parser.add_argument('--scene-split', type=str, default='all') # 'all' / 'seen' / 'unseen'
    parser.add_argument('--object-split', type=str, default='seen') # 'seen' / 'unseen'
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=128) #96
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--visualize-graph', action="store_true")
    parser.add_argument('--logging', action="store_true")
    parser.add_argument('--wandb-off', action='store_true')
    # MCTS
    parser.add_argument('--algorithm', type=str, default='mcts') # 'mcts' / 'alphago'
    parser.add_argument('--time-limit', type=int, default=None)
    parser.add_argument('--iteration-limit', type=int, default=10000)
    parser.add_argument('--max-depth', type=int, default=7)
    parser.add_argument('--rollout-policy', type=str, default='nostep') # 'nostep' / 'policy' / 'iql-policy'
    parser.add_argument('--tree-policy', type=str, default='random') # 'random' / 'policy' / 'iql-policy'
    parser.add_argument('--puct-lambda', type=float, default=0.5)
    parser.add_argument('--threshold-success', type=float, default=0.85) #0.85
    parser.add_argument('--threshold-prob', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--binary-reward', action="store_true")
    parser.add_argument('--blurring', type=int, default=3)
    parser.add_argument('--exploration', type=float, default=20) # 5 for alphago / 0.5 for mcts
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--prob-expand', type=float, default=0.5)
    # Reward model
    parser.add_argument('--normalize-reward', action="store_true")
    parser.add_argument('--reward-type', type=str, default='gt') # 'gt' / 'iql'
    parser.add_argument('--reward-model-path', type=str, default='data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--label-type', type=str, default='linspace')
    parser.add_argument('--view', type=str, default='top') 
    # Pretrained Models
    parser.add_argument('--qnet-path', type=str, default='')
    parser.add_argument('--vnet-path', type=str, default='')
    parser.add_argument('--policynet-path', type=str, default='../policy_learning/logs/0224_1815/pnet_e1.pth')
    parser.add_argument('--iql-path', type=str, default='../iql/logs/0308_0121/iql_e1.pth')
    parser.add_argument('--sigmoid', action='store_true')
    parser.add_argument('--policy-net', type=str, default='resnet') # 'resnet' / 'transport'
    parser.add_argument('--policy-version', type=int, default=-1)
    parser.add_argument('--continuous-policy', action='store_true')
    args = parser.parse_args()

    # Logger
    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    if args.logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    
    logname = 'TEST-'
    logname += args.tree_policy
    if args.tree_policy=='iql':
        logname += '_' + str(args.threshold_prob)
    if args.use_template:
        logname += '-' + args.scenes

    def print_fn(s=''):
        if args.logging: 
            logger.info(s)
            print(s)
        else: print(s)

    # Environment setup
    env = setupEnvironment(args)
    selected_objects = [['knife', 'medium'], ['fork', 'medium'], ['plate', 'medium'], ['cup', 'medium']]
    env.spawn_objects(selected_objects)
    env.arrange_objects(random=True)

    log_dir = 'data/%s' %args.algorithm
    if args.logging:
        bar = tqdm(range(args.num_scenes))
    else:
        bar = range(args.num_scenes)

    for sidx in bar:
        best_score = 0.0
        bestRgb = None
        bestRgbFront = None
        if args.logging: 
            bar.set_description("Episode %d/%d"%(sidx, args.num_scenes))
            if sidx>0:
                bar.set_postfix(success_rate="%.1f%% (%d/%d)"%(100*success/sidx, success, sidx),
                                eplen="%.1f"%(np.mean(success_eplen) if len(success_eplen)>0 else 0))
            else:
                bar.set_postfix(success_rate="0.0% (0/0)", eplen="0.0")
            
            os.makedirs('%s-%s/scene-%d'%(log_dir, log_name, sidx), exist_ok=True)
            with open('%s-%s/config.json'%(log_dir, log_name), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            logger.handlers.clear()
            formatter = logging.Formatter('%(asctime)s - %(name)s -\n%(message)s')
            file_handler = logging.FileHandler('%s-%s/scene-%d/%s.log'%(log_dir, log_name, sidx, args.algorithm))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Initial state
        obs = env.reset()
        #obs = env.get_observation() #env.reset()
        env.spawn_objects(selected_objects)
        while True:
            is_occluded = False
            is_collision = False
            env.arrange_objects(random=True)
            obs = env.get_observation()
            initRgb = obs[args.view]['rgb']
            initSeg = obs[args.view]['segmentation']
            initRgbFront = obs['front']['rgb']
            initRgbNV = obs['nv-'+args.view]['rgb']
            initSegNV = obs['nv-'+args.view]['segmentation']
            initRgbFrontNV = obs['nv-front']['rgb']
            # Check occlusions
            for o in range(len(selected_objects)):
                # get the segmentation mask of each object #
                mask = (initSeg==o+4).astype(float)
                if mask.sum()==0:
                    print_fn("Object %d is occluded."%o)
                    is_occluded = True
                    break
            # Check collision
            contact_objects = get_contact_objects()
            contact_objects = [c for c in list(get_contact_objects()) if 1 not in c and 2 not in c]
            if len(contact_objects) > 0:
                print_fn("Collision detected.")
                print_fn(contact_objects)
                is_collision = True
            if is_occluded or is_collision:
                continue
            else:
                break
        print_fn('Objects: %s' %[o for o,s in selected_objects])

        if args.logging:
            plt.imshow(initRgb)
            plt.savefig('%s-%s/scene-%d/top_initial.png'%(log_dir, log_name, sidx))
            plt.imshow(initRgbFront)
            plt.savefig('%s-%s/scene-%d/front_initial.png'%(log_dir, log_name, sidx))
            plt.imshow(initRgbNV)
            plt.savefig('%s-%s/scene-%d/nv_top_initial.png'%(log_dir, log_name, sidx))
            plt.imshow(initRgbFrontNV)
            plt.savefig('%s-%s/scene-%d/nv_front_initial.png'%(log_dir, log_name, sidx))
            plt.imshow(initSeg)
            plt.savefig('%s-%s/scene-%d/top_seg_init.png'%(log_dir, log_name, sidx))
            plt.imshow(initSegNV)
            plt.savefig('%s-%s/scene-%d/top_seg_init_nv.png'%(log_dir, log_name, sidx))

