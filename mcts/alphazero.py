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

import torch
from data_loader import TabletopTemplateDataset
from utils import loadRewardFunction, Renderer, getGraph, visualizeGraph, summaryGraph
from utils import loadIQLNetworks

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_PATH, '../..', 'TabletopTidyingUp/pybullet_ur5_robotiq'))
from custom_env import TableTopTidyingUpEnv, get_contact_objects
from utilities import Camera, Camera_front_top

countNode = {}
        
class Node(object):
    def __init__(self, numObjects, table, parent=None, preAction=None, actionProb=0.):
        self.table = table
        self.parent = parent
        self.preAction = preAction
        self.prob = actionProb

        self.numObjcts = numObjects
        if parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

        #self.isTerminal = False
        self.numVisits = 0
        self.totalReward = 0.
        self.children = {}
        # self.actionCandidates = []
        self.actionProb = None
        self.numActionCandidates = 0
        self.terminal = False

        self.reward = 0.
        self.value = 0.
        # self.qvalue = 0.
        
        if str(table) not in countNode:
            countNode[str(table)] = 1
        else:
            countNode[str(table)] += 1
    
    def takeAction(self, move):
        obj, py, px, rot = move
        posMap, rotMap = self.table
        newPosMap = copy.deepcopy(posMap)
        newRotMap = copy.deepcopy(rotMap)
        newPosMap[posMap==obj] = 0
        newPosMap[py, px] = obj
        newRotMap[posMap==obj] = 0
        newRotMap[py, px] = rot
        newTable = [newPosMap, newRotMap]
        return newTable
    
    def setActions(self, actionProb):
        self.actionProb = actionProb
        self.numActionCandidates = (actionProb>0).astype(float).sum().astype(int)
        if False: # blurring
            newActionProb = np.zeros_like(actionProb)
            for i in range(len(actionProb)):
                ap = actionProb[i]
                kernel = np.ones((2, 2))
                ap_blur = cv2.dilate(cv2.erode(ap, kernel), kernel)
                newActionProb[i] = ap_blur
            self.actionProb = newActionProb

    def isFullyExpanded(self):
        return len(self.children)!=0 and len(self.children)==self.numActionCandidates

    def __str__(self):
        s=[]
        s.append("Reward: %s"%(self.totalReward))
        s.append("Visits: %d"%(self.numVisits))
        s.append("Terminal: %s"%(self.terminal))
        s.append("Children: %d"%(len(self.children.keys())))
        return "%s: %s"%(self.__class__.__name__, ' / '.join(s))


class MCTS(object):
    def __init__(self, renderer, args, explorationConstant):
        timeLimit = args.time_limit
        iterationLimit = args.iteration_limit
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'

        self.renderer = renderer
        if self.renderer is None:
            self.domain = 'grid'
        else:
            self.domain = 'image'
        self.explorationConstant = explorationConstant
        self.puctLambda = args.puct_lambda

        self.treePolicy = args.tree_policy
        self.maxDepth = args.max_depth
        rolloutPolicy = args.rollout_policy
        if rolloutPolicy=='nostep':
            self.rollout = self.noStepPolicy
        elif rolloutPolicy=='onestep':
            self.rollout = self.oneStepPolicy
        else:
            self.rollout = lambda n: self.greedyPolicy(n, rolloutPolicy)
        self.binaryReward = args.binary_reward

        self.thresholdSuccess = args.threshold_success
        self.thresholdProb = args.threshold_prob
        self.rewardNet = None
        self.QNet = None
        self.valueNet = None
        self.policyNet = None
        self.batchSize = args.batch_size #32
        self.preProcess = None
        self.searchCount = 0
        self.blurring = args.blurring
    
    def reset(self, rgbImage, segmentation):
        table = self.renderer.setup(rgbImage, segmentation)
        self.searchCount = 0
        return table

    def setQNet(self, QNet):
        self.QNet = QNet
    
    def setRewardNet(self, rewardNet):
        self.rewardNet = rewardNet

    def setValueNet(self, valueNet):
        self.valueNet = valueNet

    def setPolicyNet(self, policyNet):
        self.policyNet = policyNet

    def setPreProcess(self, preProcess):
        self.preProcess = preProcess

    def search(self, table, needDetails=False):
        # print('search.')
        self.coverage = []
        self.root = Node(self.renderer.numObjects, table)
        if table is not None:
            self.root = Node(self.renderer.numObjects, table)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            while self.searchCount < self.searchLimit:
                self.executeRound()
        bestChild = self.getBestChild(self.root, explorationValue=0.)
        action=(action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def selectNode(self, node):
        # print('selectNode.')
        while not node.terminal: # self.isTerminal(node)[0]:
            if len(node.children)==0:
                return self.expand(node)
            elif node.isFullyExpanded() or random.uniform(0, 1) < 0.5:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def sampleFromProb(self, prob, exceptActions=[]):
        # shape: r x n x h x w
        for action in exceptActions:
            o, py, px, r = action
            prob[r-1, o-1, py, px] = 0.
        prob /= np.sum(prob)
        rs, nbs, ys, xs = np.where(prob>0.)
        assert len(rs)>0
        idx = np.random.choice(len(rs), p=prob[rs, nbs, ys, xs])
        rot, nb, y, x = rs[idx], nbs[idx], ys[idx], xs[idx]
        action = (nb+1, y, x, rot+1)
        p = prob[rot, nb, y, x]
        assert p==prob[rs, nbs, ys, xs][idx]
        return action, p
    
    def expand(self, node):
        # print('expand.')
        if node.numActionCandidates==0:
            prob = self.getPossibleActions(node)
        else:
            prob = node.actionProb
        
        if self.treePolicy=='uniform':
            prob[prob>0] = 1.
            prob /= np.sum(prob)

        exceptActions = [a for a in node.children.keys()]
        action, p = self.sampleFromProb(prob, exceptActions)

        newNode = Node(self.renderer.numObjects, node.takeAction(action), node, action, p)
        node.children[tuple(action)] = newNode
        return newNode

    def getReward(self, tables):
        # print('getReward.')
        states = []
        for table in tables:
            rgb = self.renderer.getRGB(table)
            states.append(rgb)
        s_value = torch.Tensor(np.array(states)/255.).to(torch.float32).cuda()
        s_reward = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s_reward = self.preProcess(s_reward)

        if len(states) > self.batchSize:
            rewards = []
            values = []
            numBatches = len(states)//self.batchSize
            if len(states)%self.batchSize > 0:
                numBatches += 1
            for b in range(numBatches):
                batchStatesR = s_reward[b*self.batchSize:(b+1)*self.batchSize]
                batchStatesV = [s_value[b*self.batchSize:(b+1)*self.batchSize], None, None]
                batchRewards = self.rewardNet(batchStatesR).cpu().detach().numpy()
                batchValues  = self.valueNet(batchStatesV).cpu().detach().numpy()
                rewards.append(batchRewards)
                values.append(batchValues)
            rewards = np.concatenate(rewards)
            values = np.concatenate(values)
        else:
            rewards = self.rewardNet(s_reward).cpu().detach().numpy()
            values = self.valueNet([s_value, None, None]).cpu().detach().numpy()
        return rewards.reshape(-1), values.reshape(-1)

    def backpropagate(self, node, reward, value):
        # print('backpropagate.')
        puctValue = self.puctLambda * reward + (1-self.puctLambda) * value
        while node is not None:
            node.numVisits += 1
            node.totalReward += puctValue
            node = node.parent

    def executeRound(self):
        # print('executeRound.')
        node = self.selectNode(self.root)
        reward, value= self.rollout(node)
        self.backpropagate(node, reward, value)
        self.searchCount += 1

    def getBestChild(self, node, explorationValue):
        # print('getBestChild.')
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            # print('total reward:', child.totalReward / child.numVisits)
            # print('exploration term:', explorationValue * child.prob * np.sqrt(node.numVisits) / (1 + child.numVisits))
            nodeValue = child.totalReward / child.numVisits + explorationValue * \
                child.prob * np.sqrt(node.numVisits) / (1 + child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return np.random.choice(bestNodes)

    def getPossibleActions(self, node):
        # print('getPossibleActions.')
        if node.numActionCandidates==0:
            states = []
            objectPatches = []
            for r in range(1,3):
                for o in range(self.renderer.numObjects):
                    rgbWoTarget = self.renderer.getRGB(node.table, remove=o)
                    objPatch = self.renderer.objectPatches[r][o]
                    states.append(rgbWoTarget)
                    objectPatches.append(objPatch)
            s = torch.Tensor(np.array(states)/255.).to(torch.float32).cuda()
            p = torch.Tensor(np.array(objectPatches)/255.).to(torch.float32).cuda()
            obs = [None, s, p]
            probMap = self.policyNet.get_prob(obs)
            probMap = probMap.cpu().detach().numpy()

            if self.blurring>1:
                newProbMap = np.zeros_like(probMap)
                for i in range(len(probMap)):
                    ap = probMap[i]
                    k = int(self.blurring)
                    kernel = np.ones((k, k))
                    ap_blur = cv2.dilate(ap, kernel)
                    ap_blur /= np.sum(ap_blur)
                    newProbMap[i] = ap_blur
                probMap = newProbMap

            # shape: r x n x h x w
            probMap = probMap.reshape(2, self.renderer.numObjects, self.renderer.tableSize[0], self.renderer.tableSize[1])
            probMap[probMap <= self.thresholdProb] = 0.0
            pys, pxs = np.where(node.table[0]!=0)
            for py, px in zip(pys, pxs):
                probMap[:, :, py, px] = 0
            probMap /= np.sum(probMap)
            node.setActions(probMap)

            coverageRatio = (probMap>0).sum() / probMap.reshape(-1).shape[0]
            self.coverage.append(coverageRatio)
        #print('possible actions:', len(node.actionCandidates))
        return node.actionProb
        
    def isTerminal(self, node, table=None, checkReward=False):
        # print('isTerminal')
        terminal = False
        reward = 0.0
        value = 0.0
        if table is None:
            table = node.table
        # check collision and reward
        collision = self.renderer.checkCollision(table)
        if collision:
            reward = 0.0 #-1.0
            value = 0.0 #-1.0
            terminal = True
        elif checkReward:
            reward, value = self.getReward([table])
            reward, value = reward[0], value[0]
            if reward > self.thresholdSuccess:
                terminal = True
        # check depth
        if node is not None:
            if node.depth >= self.maxDepth:
                terminal = True
            node.terminal = terminal
        return terminal, reward, value

    def noStepPolicy(self, node):
        # print('noStepPolicy.')
        # st = time.time()
        terminal, reward, value = self.isTerminal(node, checkReward=True)
        if self.binaryReward:
            if reward > self.thresholdSuccess:
                reward = 1.0
            else:
                reward = 0.0
        # et = time.time()
        # print(et - st, 'seconds.')
        return reward, value

    def oneStepPolicy(self, node):
        # print('oneStepPolicy.')
        # st = time.time()
        nb = self.renderer.numObjects
        th, tw = self.renderer.tableSize
        allPossibleActions = np.array(np.meshgrid(
                        np.arange(1, nb+1), np.arange(th), np.arange(tw), np.arange(1,3)
                        )).T.reshape(-1, 4)
        states = [self.renderer.getRGB(node.table)]
        for action in allPossibleActions:
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            states.append(self.renderer.getRGB(newNode.table))

        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = []
        numBatches = len(states)//self.batchSize
        if len(states)%self.batchSize > 0:
            numBatches += 1
        for b in range(numBatches):
            batchS = s[b*self.batchSize:(b+1)*self.batchSize]
            batchRewards = self.rewardNet(batchS).cpu().detach().numpy()
            rewards.append(batchRewards)
        rewards = np.concatenate(rewards)
        maxReward = np.max(rewards)
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward

    def greedyPolicy(self, node, policy):
        # print('greedyPolicy.')
        # st = time.time()
        c = 0
        tables = [np.copy(node.table)]
        while not (self.isTerminal(node)[0] or c>1):
            c+= 1
            if policy=='random':
                nb = self.renderer.numObjects
                th, tw = self.renderer.tableSize
                allPossibleActions = np.array(np.meshgrid(
                            np.arange(1, nb+1), np.arange(1, th-1), np.arange(1, tw-1), np.arange(1,3)
                            )).T.reshape(-1, 4)
                action = random.choice(allPossibleActions)
            else:
                if len(node.actionCandidates)==0:
                    prob = self.getPossibleActions(node, policy)
                else:
                    prob = node.actionProb
                #actions = [tuple(a) for a in actions]
                #action = random.choice(actions)
                if policy=='uniform':
                    ros, pys, pxs = np.where(prob>0)
                elif policy=='greedy':
                    ros, pys, pxs = np.where(prob==np.max(prob))
                idx = np.random.choice(len(ros))
                ro, py, px = ros[idx], pys[idx], pxs[idx]
                rot = ro // self.renderer.numObjects
                o = ro % self.renderer.numObjects
                action = (o+1, py, px, rot+1)
            
            newNode = Node(self.renderer.numObjects, node.takeAction(action), node)
            node = newNode
            # Collision check
            collision = self.renderer.checkCollision(node.table)
            if not collision:
                tables.append(np.copy(node.table))
            
        rewards, values = self.getReward(tables)
        maxReward = np.max(rewards)
        value = values[0]
        # et = time.time()
        # print(et - st, 'seconds.')
        return maxReward, value

    def greedyPolicyWithQ(self, node):
        states = [self.renderer.getRGB(node.table)]
        while not self.isTerminal(node)[0]:
            try:
                actions = self.getPossibleActions(node, 'Q')
                action = np.random.choice(actions)
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            node = newNode
            states.append(self.renderer.getRGB(node.table))
        s = torch.Tensor(np.array(states)/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        rewards = self.rewardNet(s).cpu().detach().numpy()
        maxReward = np.max(rewards)
        return maxReward

    def greedyPolicyWithV(self, node):
        state = self.renderer.getRGB(node.table)
        s = torch.Tensor(state/255.).permute([0,3,1,2]).cuda()
        if self.preProcess is not None:
            s = self.preProcess(s)
        value = self.rewardNet(s).cpu().detach().numpy()[0]
        values = [value]
        while not self.isTerminal(node)[0]:
            try:
                actions, values = self.getPossibleActions(node, 'V', needValues=True)
                idx = np.random.choice(len(actions))
                action, value = actions[idx], values[idx]
            except IndexError:
                raise Exception("Non-terminal state has no possible actions: " + str(state))
            newTable = node.takeAction(action)
            newNode = Node(self.renderer.numObjects, newTable)
            node = newNode
            values.append(value)
        maxReward = np.max(values)
        return maxReward


def setupEnvironment(objects, args):
    camera_top = Camera((0, 0, 1.45), 0.02, 2, (480, 360), 60)
    camera_front_top = Camera_front_top((0.5, 0, 1.3), 0.02, 2, (480, 360), 60)
    
    data_dir = args.data_dir
    objects_cfg = { 'paths': {
            'pybullet_object_path' : os.path.join(data_dir, 'pybullet-URDF-models/urdf_models/models'),
            'ycb_object_path' : os.path.join(data_dir, 'YCB_dataset'),
            'housecat_object_path' : os.path.join(data_dir, 'housecat6d/obj_models_small_size_final'),
        },
        'split' : 'inference' #'train'
    }
    
    gui_on = not args.gui_off
    env = TableTopTidyingUpEnv(objects_cfg, camera_top, camera_front_top, vis=gui_on, num_objs=args.num_objects, gripper_type='85')
    p.resetDebugVisualizerCamera(2.0, -270., -60., (0., 0., 0.))
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)  # Shadows on/off
    p.addUserDebugLine([0, -0.5, 0], [0, -0.5, 1.1], [0, 1, 0])

    env.reset()
    env.spawn_objects(objects)
    
    env.arrange_objects(random=True)
    return env


if __name__=='__main__':
    parser = ArgumentParser()
    # Data directory
    parser.add_argument('--data-dir', type=str, default='/ssd/disk')
    # Inference
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument('--num-objects', type=int, default=4)
    parser.add_argument('--num-scenes', type=int, default=10)
    parser.add_argument('--H', type=int, default=12)
    parser.add_argument('--W', type=int, default=15)
    parser.add_argument('--crop-size', type=int, default=128) #96
    parser.add_argument('--gui-off', action="store_true")
    parser.add_argument('--visualize-graph', action="store_true")
    parser.add_argument('--logging', action="store_true")
    # MCTS
    parser.add_argument('--time-limit', type=int, default=None)
    parser.add_argument('--iteration-limit', type=int, default=10000)
    parser.add_argument('--max-depth', type=int, default=7)
    parser.add_argument('--rollout-policy', type=str, default='nostep') # 'greedy' / 'uniform'
    parser.add_argument('--tree-policy', type=str, default='p') # 'p' / 'uniform
    parser.add_argument('--puct-lambda', type=float, default=0.5)
    parser.add_argument('--puct-constant', type=float, default=5)
    parser.add_argument('--policy-net', type=str, default='resnet') # 'resnet' / 'transport'
    parser.add_argument('--threshold-success', type=float, default=0.9) #0.85
    parser.add_argument('--threshold-prob', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--binary-reward', action="store_true")
    parser.add_argument('--blurring', type=int, default=1)
    # Reward model
    parser.add_argument('--reward-model-path', type=str, default='data/classification-best/top_nobg_linspace_mse-best.pth')
    parser.add_argument('--label-type', type=str, default='linspace')
    parser.add_argument('--view', type=str, default='top') 
    # Pretrained Models
    parser.add_argument('--iql-path', type=str, default='../iql/logs/0308_0121/iql_e1.pth')
    args = parser.parse_args()

    # Logger
    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    if args.logging:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    def print_fn(s=''):
        if args.logging: logger.info(s)
        else: print(s)

    # Random seed
    seed = args.seed
    if seed is not None:
        print_fn("Random seed: %d"%seed)
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Environment setup
    objects = ['bowl', 'can_drink', 'plate', 'marker', 'soap_dish', 'book', 'remote', 'fork', 'knife', 'spoon', 'teapot', 'cup']
    objects = [(o, 'medium') for o in objects]
    #objects = [('bowl', 'medium'), ('can_drink','medium'), ('plate','medium'), ('marker', 'medium'), ('soap_dish', 'medium'), ('book', 'medium'), ('remote', 'medium')]
    selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
    env = setupEnvironment(selected_objects, args)

    # MCTS setup
    renderer = Renderer(tableSize=(args.H, args.W), imageSize=(360, 480), cropSize=(args.crop_size, args.crop_size))
    searcher = MCTS(renderer, args, explorationConstant=args.puct_constant)

    # Network setup
    model_path = args.reward_model_path
    rewardNet, preprocess = loadRewardFunction(model_path)
    searcher.setRewardNet(rewardNet)
    searcher.setPreProcess(preprocess)

    # IQL policy
    pnet, valuenet = loadIQLNetworks(args.iql_path, args)
    pnet = pnet.to("cuda:0")
    valuenet = valuenet.to("cuda:0")
    searcher.setPolicyNet(pnet)
    searcher.setValueNet(valuenet)

    success = 0
    log_dir = 'data/alphago'
    if args.logging:
        bar = tqdm(range(args.num_scenes))
    else:
        bar = range(args.num_scenes)

    for sidx in bar:
        if args.logging: 
            bar.set_description("Episode %d/%d"%(sidx, args.num_scenes))
            if sidx>0:
                bar.set_postfix(success_rate="%.1f%% (%d/%d)"%(100*success/sidx, success, sidx))
            else:
                bar.set_postfix(success_rate="0.0% (0/0)")
            
            os.makedirs('%s-%s/scene-%d'%(log_dir, log_name, sidx), exist_ok=True)
            with open('%s-%s/config.json'%(log_dir, log_name), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

            logger.handlers.clear()
            formatter = logging.Formatter('%(asctime)s - %(name)s -\n%(message)s')
            file_handler = logging.FileHandler('%s-%s/scene-%d/alphago.log'%(log_dir, log_name, sidx))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if seed is not None: 
            np.random.seed(seed + sidx)
        
        # Initial state
        obs = env.reset()
        selected_objects = [objects[i] for i in np.random.choice(len(objects), args.num_objects, replace=False)]
        env.spawn_objects(selected_objects)
        while True:
            is_occluded = False
            is_collision = False
            env.arrange_objects(random=True)
            obs = env.get_observation()
            initRgb = obs[args.view]['rgb']
            initSeg = obs[args.view]['segmentation']
            # Check occlusions
            for o in range(args.num_objects):
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

        if args.logging:
            plt.imshow(initRgb)
            plt.savefig('%s-%s/scene-%d/initial.png'%(log_dir, log_name, sidx))
        initTable = searcher.reset(initRgb, initSeg)
        print_fn('initTable: \n %s' % initTable[0])
        table = initTable

        print_fn("--------------------------------")
        for step in range(10):
            st = time.time()
            countNode = {}
            resultDict = searcher.search(table=table, needDetails=True)
            print_fn("Num Children: %d"%len(searcher.root.children))
            for i, c in enumerate(sorted(list(searcher.root.children.keys()))):
                print_fn(f"{i} {c} {str(searcher.root.children[c])}")
            action = resultDict['action']
            et = time.time()
            print_fn(f'{et-st} seconds to search.')
            print_fn("Action coverage: %f"%np.mean(searcher.coverage))

            summary = summaryGraph(searcher.root)
            if args.visualize_graph:
                graph = getGraph(searcher.root)
                fig = visualizeGraph(graph, title='AlphaGo')
                fig.show()
            print_fn(summary)

            # action probability
            actionProb = searcher.root.actionProb
            if args.logging and actionProb is not None:
                if args.tree_policy=='uniform':
                    actionProb[actionProb>0] = 1.
                    actionProb /= np.sum(actionProb)
                actionProb[actionProb>args.threshold_prob] += 0.5
                plt.imshow(np.mean(actionProb, axis=(0, 1)))
                plt.savefig('%s-%s/scene-%d/actionprob_%d.png'%(log_dir, log_name, sidx, step))

            # expected result in mcts #
            nextTable = searcher.root.takeAction(action)
            print_fn("Best Action: %s"%str(action))
            print_fn("Best Child: \n %s"%nextTable[0])

            nextCollision = renderer.checkCollision(nextTable)
            print_fn("Collision: %s"%nextCollision)
            print_fn("Save fig: scene-%d/expect_%d.png"%(sidx, step))
            tableRgb = renderer.getRGB(nextTable)
            if args.logging:
                plt.imshow(tableRgb)
                plt.savefig('%s-%s/scene-%d/expect_%d.png'%(log_dir, log_name, sidx, step))

            # simulation step in pybullet #
            target_object, target_position, rot_angle = renderer.convert_action(action)
            obs = env.step(target_object, target_position, rot_angle)
            currentRgb = obs[args.view]['rgb']
            currentSeg = obs[args.view]['segmentation']
            if args.logging:
                plt.imshow(currentRgb)
                plt.savefig('%s-%s/scene-%d/real_%d.png'%(log_dir, log_name, sidx, step))

            table = searcher.reset(currentRgb, currentSeg)
            if table is None:
                print_fn("Scenario ended.")
                break
            #table = copy.deepcopy(nextTable)
            print_fn("Current state: \n %s"%table[0])

            terminal, reward, _ = searcher.isTerminal(None, table, checkReward=True)
            print_fn("Current Score: %f" %reward)
            print_fn("--------------------------------")
            if terminal:
                print_fn("Arrived at the final state:")
                print_fn("Score: %f"%reward)
                if reward > args.threshold_success:
                    success += 1
                print_fn(table[0])
                print_fn("--------------------------------")
                print_fn("--------------------------------")
                if args.logging:
                    plt.imshow(currentRgb)
                    plt.savefig('%s-%s/scene-%d/final.png'%(log_dir, log_name, sidx))
                break
    print_fn("Success rate: %.2f (%d/%d)"%(success/args.num_scenes, success, args.num_scenes))
    print("Success rate: %.2f (%d/%d)"%(success/args.num_scenes, success, args.num_scenes))

