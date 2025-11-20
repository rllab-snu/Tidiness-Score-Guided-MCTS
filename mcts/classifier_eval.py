import argparse
import copy
import os
import numpy as np
from tqdm import tqdm
from data_loader import PybulletNpyDataset, TabletopTemplateDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50


preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def eval(args):
    batch_size = args.batch_size
    save_dir = os.path.join('data', args.out)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # dataloader #
    print("Loading data...")

    test_su_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-seen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view)
    test_us_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-seen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view)
    test_uu_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view)
    print('len(test_su_dataset):', len(test_su_dataset))
    print('len(test_us_dataset):', len(test_us_dataset))
    print('len(test_uu_dataset):', len(test_uu_dataset))

    test_su_dataloader = DataLoader(test_su_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_us_dataloader = DataLoader(test_us_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_uu_dataloader = DataLoader(test_uu_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    # model #
    print("Create a ResNet model.")
    if args.model=='resnet-18':
        resnet = resnet18
    elif args.model=='resnet-34':
        resnet = resnet34
    elif args.model=='resnet-50':
        resnet = resnet50

    # replace the last fc layer #
    model = resnet(pretrained=False)
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
        #nn.Sigmoid()
    )
    model.load_state_dict(torch.load(args.model_path))
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)
    model.eval()

    accuracies = []
    matchings = []
    for X_val, Y_val in tqdm(test_su_dataloader):
        X_val = preprocess(X_val).to(device)
        Y_val = Y_val[:, 0].to(device)
        y_pred = model(X_val)[:, 0]
        indices = torch.logical_or(Y_val==0, Y_val==1)
        matching = (y_pred.round()==Y_val)[indices].float().detach().cpu().numpy()
        matchings.append(matching)
    matchings = np.concatenate(matchings, axis=0)
    accuracy = np.mean(matchings)
    print("Test SU accuracy:", accuracy)
    accuracies.append(accuracy)

    matchings = []
    for X_val, Y_val in tqdm(test_us_dataloader):
        X_val = preprocess(X_val).to(device)
        Y_val = Y_val[:, 0].to(device)
        y_pred = model(X_val)[:, 0]
        indices = torch.logical_or(Y_val == 0, Y_val == 1)
        matching = (y_pred.round() == Y_val)[indices].float().detach().cpu().numpy()
        matchings.append(matching)
    matchings = np.concatenate(matchings, axis=0)
    accuracy = np.mean(matchings)
    print("Test US accuracy:", accuracy)
    accuracies.append(accuracy)

    matchings = []
    for X_val, Y_val in tqdm(test_uu_dataloader):
        X_val = preprocess(X_val).to(device)
        Y_val = Y_val[:, 0].to(device)
        y_pred = model(X_val)[:, 0]
        indices = torch.logical_or(Y_val == 0, Y_val == 1)
        matching = (y_pred.round() == Y_val)[indices].float().detach().cpu().numpy()
        matchings.append(matching)
    matchings = np.concatenate(matchings, axis=0)
    accuracy = np.mean(matchings)
    print("Test UU accuracy:", accuracy)
    accuracies.append(accuracy)

    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluation
    parser.add_argument("--batch_size", type=int, default=16) #100
    parser.add_argument("--out", type=str, default='classification')
    parser.add_argument("--gpu", type=int, default=0)
    # Dataset
    # parser.add_argument("--dataset", type=str, default='tabletop')
    parser.add_argument("--data_dir", type=str, default='/ssd/disk/TableTidyingUp/dataset_template')
    #parser.add_argument("--data_dir", type=str, default='/ssd/disk/ur5_tidying_data/pybullet_single_bg')
    parser.add_argument("--remove_bg", action="store_true") # default: False
    parser.add_argument("--label_type", type=str, default='linspace') # linspace / binary
    parser.add_argument("--view", type=str, default='top') # top / front_top
    # etc
    parser.add_argument("--model", type=str, default='resnet-18')
    parser.add_argument("--model_path", type=str, default='data/classification/top_nobg_linspace_mse-best.pth')
    args = parser.parse_args()

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print("Evaluation starts.")
    accuracies = eval(args)
    print("Evaluation finished.")
    print("Average accuracy:", float(np.mean(accuracies)))

