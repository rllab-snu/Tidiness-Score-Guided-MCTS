import cv2
import os
import sys
import numpy as np
import igraph as ig
import plotly.graph_objects as go
from igraph import Graph, EdgeSeq
from ellipse import LsqEllipse
from contextlib import contextmanager

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

def summaryGraph(root):
    countDepth = {}
    visitDepth = {}
    def countSubGraph(node):
        assert (node.terminal and len(node.children)==0) or not node.terminal
        if node.depth not in countDepth:
            countDepth[node.depth] = 0
            visitDepth[node.depth] = 0
        countDepth[node.depth] += 1
        visitDepth[node.depth] += node.numVisits
        if len(node.children)==0:
            return
        for c in node.children:
            countSubGraph(node.children[c])
    countSubGraph(root)

    txt = ""
    txt += "Total nodes: %d\n"%(sum(countDepth.values()))
    for d in countDepth:
        txt += "Depth %d: %d nodes, %d visits\n"%(d, countDepth[d], visitDepth[d])
    return txt

def getGraph(root):
    g = ig.Graph(n=1)

    def getSubGraph(node, vNode):
        g.vs[vNode]['visit'] = node.numVisits
        g.vs[vNode]['reward'] = node.totalReward
        g.vs[vNode]['depth'] = node.depth
        vertices = 0
        edges = []
        if len(node.children)==0:
            return vertices, edges
        for c in node.children:
            vChild = g.vcount()
            eChild = g.ecount()

            vertices += 1
            edges.append([vNode, vChild])
            g.add_vertices(1)
            g.add_edges([(vNode, vChild)])

            # g.vs[vChild]['visit'] = node.children[c].numVisits
            # g.vs[vChild]['reward'] = node.children[c].totalReward
            g.es[eChild]['action'] = str(c)

            childNode = node.children[c]
            childV, childE = getSubGraph(childNode, vChild)

            vertices += childV
            edges += childE
        return vertices, edges

    v, e = getSubGraph(root, 0)
    #g2 = ig.Graph(n=v+1, edges=e)
    return g

def visualizeGraph(graph, title):
    nr_vertices = graph.vcount()
    v_label = ["%d"%(v['visit']) for v in graph.vs]
    r_label = ["%.2f"%(v['reward']) for v in graph.vs]
    labels = ["visits: %d / rewards: %.2f / depth: %d"%(v['visit'], v['reward'], v['depth']) for v in graph.vs]
    #v_label = list(map(str, range(nr_vertices)))
    lay = graph.layout('rt') #Graph.layout_reingold_tilford_circular

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    E = [e.tuple for e in graph.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    edge_label = []
    edge_position = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

        edge_position.append([(position[edge[0]][0]+position[edge[1]][0])/2, (position[edge[0]][1]+position[edge[1]][1])/2])
        action = graph.es.find(_source=edge[0], _target=edge[1])['action'] 
        if len(action)>1: action = ''
        edge_label.append(action)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=Xe,
                    y=Ye,
                    mode='lines',
                    line=dict(color='rgb(210,210,210)', width=1),
                    hoverinfo='none'
                    ))
    fig.add_trace(go.Scatter(
                    x=Xn,
                    y=Yn,
                    mode='markers',
                    name='bla',
                    marker=dict(symbol='circle', # square / circle-dot
                                size=18,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                    text=labels,
                    hoverinfo='text',
                    opacity=0.8
                    ))

    axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )
    
    def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
        L=len(pos)
        if len(text)!=L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=text[k], # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2*M-pos[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

    fig.update_layout(title=title,
              annotations=make_annotations(list(position.values())+edge_position, v_label+edge_label, font_size=8, font_color='rgb(10,10,10)'),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )
    return fig


def loadPolicyNetwork(model_path, args):
    sys.path.append(os.path.join(FILE_PATH, '..', 'policy_learning'))
    from model import ResNet
    pnet = ResNet()
    pnet.load_state_dict(torch.load(model_path))
    return pnet

def loadIQLPolicyNetwork(model_path, args):
    sys.path.append(os.path.join(FILE_PATH, '..', 'iql'))
    if args.policy_version!=-1:
        if args.policy_version==0:
            from src.policy import PolicyOpt0
            policy = PolicyOpt0()
        elif args.policy_version==1:
            from src.policy import PolicyOpt1
            policy = PolicyOpt1()
        elif args.policy_version==2:
            from src.policy import PolicyOpt2
            policy = PolicyOpt2()
        elif args.policy_version==3:
            from src.policy import PolicyOpt3
            policy = PolicyOpt3()
        elif args.policy_version==4:
            from src.policy import PolicyOpt4
            policy = PolicyOpt4()
    elif args.continuous_policy:
        from src.policy import GaussianPolicy
        policy = GaussianPolicy()
    else:
        if args.policy_net=='transport':
            from src.policy import DiscreteTransportPolicy
            policy = DiscreteTransportPolicy(crop_size=args.crop_size)
        elif args.policy_net=='resnet':
            from src.policy import DiscreteResNetPolicy
            policy = DiscreteResNetPolicy(crop_size=args.crop_size)
    state_dict = torch.load(model_path)
    if type(state_dict)==dict:
        policy.load_state_dict(state_dict['policy_state_dict'])
    else:
        state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
        policy.load_state_dict(state_dict)
    return policy

def loadIQLRewardNetwork(model_path, args, sigmoid=False):
    sys.path.append(os.path.join(FILE_PATH, '..', 'iql'))
    from src.value_functions import RewardFunction
    rewardNet = RewardFunction(hidden_dim=256, sigmoid=sigmoid)
    state_dict = torch.load(model_path)
    state_dict = {k.replace('rf.', ''): v for k, v in state_dict.items() if k.startswith('rf.')}
    rewardNet.load_state_dict(state_dict)
    return rewardNet

def loadIQLValueNetwork(model_path, args):
    sys.path.append(os.path.join(FILE_PATH, '..', 'iql'))
    from src.value_functions import ValueFunction
    valueNet = ValueFunction(hidden_dim=256)
    state_dict = torch.load(model_path)
    state_dict = {k.replace('vf.', ''): v for k, v in state_dict.items() if k.startswith('vf.')}
    valueNet.load_state_dict(state_dict)
    return valueNet

def loadIQLNetworks(model_path, args):
    state_dict = torch.load(model_path)
    sys.path.append(os.path.join(FILE_PATH, '..', 'iql'))
    # policy
    if args.policy_net=='transport':
        from src.policy import DiscreteTransportPolicy
        policy = DiscreteTransportPolicy(crop_size=args.crop_size)
    elif args.policy_net=='resnet':
        from src.policy import DiscreteResNetPolicy
        policy = DiscreteResNetPolicy(crop_size=args.crop_size)
    policy_state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items() if k.startswith('policy.')}
    policy.load_state_dict(policy_state_dict)
    # value function
    from src.value_functions import ValueFunction
    valueNet = ValueFunction(hidden_dim=256)
    value_state_dict = {k.replace('vf.', ''): v for k, v in state_dict.items() if k.startswith('vf.')}
    valueNet.load_state_dict(value_state_dict)
    # reward function
    from src.value_functions import RewardFunction
    reward_state_dict = {k.replace('rf.', ''): v for k, v in state_dict.items() if k.startswith('rf.')}
    if len(reward_state_dict)>0:
        rewardNet = RewardFunction(hidden_dim=256, sigmoid=args.sigmoid)
        rewardNet.load_state_dict(reward_state_dict)
    else:
        rewardNet = None
    return policy, valueNet, rewardNet

def loadRewardFunction(model_path):
    vNet = resnet18(pretrained=False)
    fc_in_features = vNet.fc.in_features
    vNet.fc = nn.Sequential(nn.Linear(fc_in_features, 1))
    vNet.load_state_dict(torch.load(model_path))
    vNet.to("cuda:0")
    vNet.eval()
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vNet, preprocess

class Renderer(object):
    def __init__(self, tableSize, imageSize, cropSize):
        self.tableSize = np.array(tableSize)
        self.imageSize = np.array(imageSize)
        self.cropSize = np.array(cropSize)
        self.rgb = None

    def setup(self, rgbImage, segmentation, numRotations=2):
        # segmentation info
        # 1: background (table)
        # 2: None
        # 3: robot arm
        # 4~N: objects
        self.numObjects = int(np.max(segmentation)) - 3
        self.ratio, self.offset = self.getRatio()
        self.masks, self.centers = self.getMasks(segmentation)
        if np.isnan(self.centers).any():
            return None
        self.rgb = np.copy(rgbImage)
        oPatches, oMasks = self.getObjectPatches()
        self.objectPatches, self.objectMasks, self.objectAngles = self.getRotatedPatches(oPatches, oMasks, numRotations)
        self.segmap = np.copy(segmentation)
        posMap = self.getTable(segmentation)
        rotMap = np.zeros_like(posMap)
        table = [posMap, rotMap]
        return table

    def getRatio(self):
        # v2.
        ratio = self.imageSize / self.tableSize
        offset = 0.0
        # ty, tx = np.round((np.array([py, px]) + 0.5) * ratio - 0.5).astype(int)
        # gy, gx = np.round((np.array(center) + 0.5) / ratio - 0.5).astype(int)

        # v1.
        # ratio = self.imageSize // self.tableSize
        # offset = (self.imageSize - ratio * self.tableSize + ratio)//2
        # ty, tx = np.array([py, px]) * self.ratio + self.offset
        # gy, gx = ((np.array(center) - self.offset) // self.ratio).astype(int)
        return ratio, offset
    
    def getMasks(self, segmap):
        masks, centers = [], []
        for o in range(self.numObjects):
            # get the segmentation mask of each object #
            mask = (segmap==o+4).astype(float)
            # if mask.sum()<100:
            #     kernel = np.ones((2, 2), np.uint8)
            #     mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
            # else:
            #     kernel = np.ones((3, 3), np.uint8)
            #     mask = cv2.dilate(cv2.erode(mask, kernel), kernel)
            mask = np.round(mask)
            masks.append(mask)
            # get the center of each object #
            py, px = np.where(mask)
            cy, cx = np.round([np.mean(py), np.mean(px)])
            center = (cy, cx)
            centers.append(center)
        return masks, centers
    
    def getObjectPatches(self):
        objPatches = []
        objMasks = []
        for o in range(self.numObjects):
            mask = self.masks[o]
            cy, cx = self.centers[o]

            yMin = int(cy-self.cropSize[0]/2)
            yMax = int(cy+self.cropSize[0]/2)
            xMin = int(cx-self.cropSize[1]/2)
            xMax = int(cx+self.cropSize[1]/2)
            objPatch = np.zeros([*self.cropSize, 3])
            objPatch[
                max(0, -yMin): max(0, -yMin) + min(self.imageSize[0], yMax) - max(0, yMin),
                max(0, -xMin): max(0, -xMin) + min(self.imageSize[1], xMax) - max(0, xMin),
            ] = self.rgb[
                    max(0, yMin): min(self.imageSize[0], yMax),
                    max(0, xMin): min(self.imageSize[1], xMax),
                    :3
                ] * mask[
                        max(0, yMin): min(self.imageSize[0], yMax),
                        max(0, xMin): min(self.imageSize[1], xMax)
                    ][:, :, None]

            objMask = np.zeros(self.cropSize)
            objMask[
                max(0, -yMin): max(0, -yMin) + min(self.imageSize[0], yMax) - max(0, yMin),
                max(0, -xMin): max(0, -xMin) + min(self.imageSize[1], xMax) - max(0, xMin)
            ] = mask[
                    max(0, yMin): min(self.imageSize[0], yMax),
                    max(0, xMin): min(self.imageSize[1], xMax)
                ]
            # plt.imshow(objPatch/255.)
            # plt.show()
            objPatches.append(objPatch)
            objMasks.append(objMask)
        return objPatches, objMasks

    def getRotatedPatches(self, objPatches, objMasks, numRotations=2):
        rotatedObjPatches = [[] for _ in range(numRotations)]
        rotatedObjMasks = [[] for _ in range(numRotations)]
        rotatedAngles = [[] for _ in range(numRotations)]
        for o in range(len(objPatches)):
            patch = objPatches[o]
            mask = objMasks[o]
            py, px = np.where(mask)
            cy, cx = np.round([np.mean(py), np.mean(px)])
            X = np.array(list(zip(px, py)))
            if len(X) < 5:
                # can be a rectangle
                rect = cv2.minAreaRect(X)
                phi = rect[2] + 90
            else:
                try:
                    reg = LsqEllipse().fit(X)
                    center, width, height, phi = reg.as_parameters()
                    phi = phi * 180 / np.pi
                    if np.abs(width-height) < 6:
                        # can be a rectangle
                        rect = cv2.minAreaRect(X)
                        phi = rect[2] + 90
                except:
                    rect = cv2.minAreaRect(X)
                    phi = rect[2] + 90
            for r in range(numRotations):
                angle = phi + r * 180 / numRotations
                #angle = phi / np.pi * 180 + r * 90
                height, width = mask.shape[:2]
                matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                patch_rotated = cv2.warpAffine(patch.copy(), matrix, (width, height))
                mask_rotated = cv2.warpAffine(mask.copy(), matrix, (width, height))
                rotatedObjPatches[r].append(patch_rotated)
                rotatedObjMasks[r].append(mask_rotated)
                rotatedAngles[r].append(angle)

        objectPatches = [objPatches] + rotatedObjPatches
        objectMasks = [objMasks] + rotatedObjMasks
        objectAngles = [[0. for _ in range(len(objPatches))]] + rotatedAngles
        if False: # check patches
            for r in range(len(objectPatches)):
                for o in range(len(objectPatches[r])):
                    plt.imshow(objectPatches[r][o]/255.)
                    plt.savefig('data/mcts-ellipse/%d_%d.png'%(o, r))
        return objectPatches, objectMasks, objectAngles

    def getRGB(self, table, remove=None):
        posMap, rotMap = table
        newRgb = np.zeros_like(np.array(self.rgb)[:, :, :3])
        for o in range(self.numObjects):
            if remove is not None:
                if o==remove:
                    continue
            if (posMap==o+1).any():
                py, px = np.where(posMap==o+1)
                py, px = py[0], px[0]
                ty, tx = np.round((np.array([py, px]) + 0.5) * self.ratio - 0.5).astype(int)
                # ty, tx = np.array([py, px]) * self.ratio + self.offset
                rot = int(rotMap[py, px])
            else:
                ty, tx = self.centers[o]
                rot = 0
            yMin = int(ty - self.cropSize[0] / 2)
            yMax = int(ty + self.cropSize[0] / 2)
            xMin = int(tx - self.cropSize[1] / 2)
            xMax = int(tx + self.cropSize[1] / 2)
            newRgb[
                max(0, yMin): min(self.imageSize[0], yMax),
                max(0, xMin): min(self.imageSize[1], xMax)
            ] += (self.objectPatches[rot][o] * self.objectMasks[rot][o][:, :, None])[
                    max(0, -yMin): max(0, -yMin) + (min(self.imageSize[0], yMax) - max(0, yMin)),
                    max(0, -xMin): max(0, -xMin) + (min(self.imageSize[1], xMax) - max(0, xMin)),
                ].astype(np.uint8)
        # plt.imshow(newRgb)
        # plt.show()
        return np.array(newRgb)
    
    def checkCollision(self, table):
        posMap, rotMap = table
        collisionMask = np.zeros([*np.array(self.rgb).shape[:2]])
        for o in range(self.numObjects):
            if (posMap==o+1).any():
                py, px = np.where(posMap==o+1)
                py, px = py[0], px[0]
                ty, tx = np.round((np.array([py, px]) + 0.5) * self.ratio - 0.5).astype(int)
                # ty, tx = np.array([py, px]) * self.ratio + self.offset
                rot = int(rotMap[py, px])
            else:
                ty, tx = self.centers[o]
                rot = 0
            yMin = int(ty - self.cropSize[0] / 2)
            yMax = int(ty + self.cropSize[0] / 2)
            xMin = int(tx - self.cropSize[1] / 2)
            xMax = int(tx + self.cropSize[1] / 2)
            collisionMask[
                max(0, yMin): min(self.imageSize[0], yMax),
                max(0, xMin): min(self.imageSize[1], xMax)
            ] += self.objectMasks[rot][o][
                    max(0, -yMin): max(0, -yMin) + (min(self.imageSize[0], yMax) - max(0, yMin)),
                    max(0, -xMin): max(0, -xMin) + (min(self.imageSize[1], xMax) - max(0, xMin)),
                ]
        if (collisionMask>1).any():
            return True
        else:
            return False

    def getTable(self, segmap):
        newTable = np.zeros([self.tableSize[0], self.tableSize[1]])
        # return newTable
        for o in range(self.numObjects):
            center = self.centers[o]
            gyx = (np.array(center) + 0.5) / self.ratio - 0.5
            if np.linalg.norm(gyx - np.round(gyx))<0.2:
                gy, gx = np.round(gyx).astype(int)
                # gy, gx = np.round((np.array(center) + 0.5) / self.ratio - 0.5).astype(int)
                # gy, gx = ((np.array(center) - self.offset) // self.ratio).astype(int)
                newTable[gy, gx] = o + 1
        return newTable

    def convert_action(self, action):
        obj, py, px, rot = action
        target_object = obj + 3
        ty, tx = np.round((np.array([py, px]) + 0.5) * self.ratio - 0.5).astype(int)
        # ty, tx = np.array([py, px]) * self.ratio + self.offset
        target_position = [ty, tx]

        rot_angle = self.objectAngles[rot][obj-1]
        rot_angle = rot_angle / 180 * np.pi
        return target_object, target_position, rot_angle
