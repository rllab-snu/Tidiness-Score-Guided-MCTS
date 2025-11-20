import argparse
import copy
import os
import numpy as np
import datetime
import wandb
from tqdm import tqdm
from data_loader import TabletopTemplateDataset

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1):
        super(TwoLayerMLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def train(args, log_name):
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    lrate = args.lr
    save_dir = os.path.join('data', args.out, log_name)
    os.makedirs(save_dir, exist_ok=True)
    save_model = True #False
    save_freq = args.save_freq

    # dataloader #
    print("Loading data...")
    dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'train'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, get_mask=True)
    us_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-seen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, get_mask=True)
    su_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-seen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, get_mask=True)
    uu_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, get_mask=True)
    print('len(train_dataset):', len(dataset))
    print('len(us_test_dataset):', len(us_test_dataset))
    print('len(su_test_dataset):', len(su_test_dataset))
    print('len(uu_test_dataset):', len(uu_test_dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    us_test_dataloader = DataLoader(us_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    su_test_dataloader = DataLoader(su_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    uu_test_dataloader = DataLoader(uu_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # model #
    print("Create a MLP model.")
    model_name = 'fft_' + args.view.replace('_', '')
    model_name += '_nobg_' if args.remove_bg else '_bg_'
    model_name += args.label_type + '_'
    model_name += args.loss
    
    if args.model.startswith('resnet'):
        if args.model=='resnet-18':
            resnet = resnet18
        elif args.model=='resnet-34':
            resnet = resnet34
        elif args.model=='resnet-50':
            resnet = resnet50
        model = resnet(pretrained=False)
        fc_in_features = model.fc.in_features
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Sequential(
            nn.Linear(fc_in_features, 1),
            nn.Sigmoid()
        )
    else:
        model = TwoLayerMLP(input_size=36+48, hidden_size=256, output_size=1)
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model.to(device)

    # loss function and optimizer #
    if args.loss=='mse':
        loss_fn = nn.MSELoss()
    elif args.loss=='bce':
        loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
 
    # Hold the best model
    best_accuracy = - np.inf   # init to negative infinity
    best_weights = None
    
    resize = None
    # resize = torchvision.transforms.Resize([36, 48])
    mean = 50
    std = 20

    count_steps = 0
    for epoch in range(n_epoch):
        model.train()
        with tqdm(dataloader) as bar:
            bar.set_description(f"Epoch {epoch}")
            for X_batch, Y_batch, Z_batch in bar:
                # FFT2
                if resize is None:
                    X_resized = X_batch.numpy().astype(bool).astype(float)
                else:
                    X_resized = resize(X_batch).numpy().astype(bool).astype(float)
                f = np.fft.fft2(X_resized)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20*np.log(np.abs(fshift+1e-8))
                ms_norm = (magnitude_spectrum - mean) / std
                # ms_norm = (magnitude_spectrum - magnitude_spectrum.mean()) / magnitude_spectrum.std()
                assert not np.isnan(ms_norm).any()
                if args.model.startswith('resnet'):
                    M_batch = torch.Tensor(ms_norm[:, None, :]).to(X_batch.dtype).to(device)
                else:
                    ms_x = ms_norm.max(1)
                    ms_y = ms_norm.max(2)
                    ms_concat = np.concatenate([ms_x, ms_y], 1)
                    M_batch = torch.Tensor(ms_concat).to(X_batch.dtype).to(device)

                Y_batch = Y_batch[:, 0].to(device)
                # forward pass
                y_pred = model(M_batch)[:, 0]
                loss = loss_fn(y_pred, Y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                indices = torch.logical_or(Y_batch==0, Y_batch==1)
                acc = (y_pred.round() == Y_batch)[indices].float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
                if not args.wandb_off:
                    step_log = {'train loss': float(loss), 'train accuracy': float(acc)}
                    wandb.log(step_log)
                count_steps += 1

        # evaluate accuracy at end of each epoch
        model.eval()
        accuracies = []
        for test_dataloader in [us_test_dataloader, su_test_dataloader, uu_test_dataloader]:
            matchings = []
            for X_val, Y_val, Z_val in test_dataloader:
                # FFT2
                if resize is None:
                    X_resized = X_val.numpy().astype(bool).astype(float)
                else:
                    X_resized = resize(X_val).numpy().astype(bool).astype(float)
                f = np.fft.fft2(X_resized)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = 20*np.log(np.abs(fshift + 1e-8))
                ms_norm = (magnitude_spectrum - mean) / std
                assert not np.isnan(ms_norm).any()
                if args.model.startswith('resnet'):
                    M_val = torch.Tensor(ms_norm[:, None, :]).to(X_val.dtype).to(device)
                else:
                    ms_x = ms_norm.max(1)
                    ms_y = ms_norm.max(2)
                    ms_concat = np.concatenate([ms_x, ms_y], 1)
                    M_val = torch.Tensor(ms_concat).to(X_val.dtype).to(device)

                Y_val = Y_val[:, 0].to(device)
                y_pred = model(M_val)[:, 0]
                indices = torch.logical_or(Y_val==0, Y_val==1)
                matching = (y_pred.round()==Y_val)[indices].float().detach().cpu().numpy()
                matchings.append(matching)
            matchings = np.concatenate(matchings, axis=0)
            accuracy = np.mean(matchings)
            accuracies.append(accuracy)
        print("US validation accuracy:", accuracies[0])
        print("SU validation accuracy:", accuracies[1])
        print("UU validation accuracy:", accuracies[2])
        print("mean validation accuracy:", np.mean(accuracies))
        if not args.wandb_off:
            step_log = {
                    'us valid accuracy': float(accuracies[0]),
                    'su valid accuracy': float(accuracies[1]),
                    'uu valid accuracy': float(accuracies[2]),
                    'mean valid accuracy': float(np.mean(accuracies)),
                    }
            wandb.log(step_log)
        if np.mean(accuracies) > best_accuracy:
            best_accuracy = np.mean(accuracies)
            best_weights = copy.deepcopy(model.state_dict())

        # optionally save model
        if save_model and (epoch+1)%save_freq==0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}-{epoch}.pth"))
            print('saved model at ' + os.path.join(save_dir, f"{model_name}-{epoch}.pth"))

    # restore model and return best accuracy
    #model.load_state_dict(best_weights)
    if save_model:
        torch.save(best_weights, os.path.join(save_dir, f"{model_name}-best.pth"))
        print('saved model at ' + os.path.join(save_dir, f"{model_name}-best.pth"))
    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16) #100
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default='classification')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--save_freq", type=int, default=1) #5
    # Dataset
    parser.add_argument("--data_dir", type=str, default='/ssd/disk/TableTidyingUp/dataset_template')
    parser.add_argument("--remove_bg", action="store_true") # default: False
    parser.add_argument("--label_type", type=str, default='binary') # linspace / binary
    parser.add_argument("--view", type=str, default='top') # top / front_top
    # etc
    parser.add_argument("--model", type=str, default='resnet-18')
    parser.add_argument('--wandb-off', action='store_true')
    args = parser.parse_args()

    now = datetime.datetime.now()
    log_name = now.strftime("%m%d_%H%M")
    if not args.wandb_off:
        wandb.init(project="Classifier")
        wandb.config.update(parser.parse_args())
        wandb.run.name = log_name
        wandb.run.save()

    gpu = args.gpu
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if str(gpu) in visible_gpus:
            gpu_idx = visible_gpus.index(str(gpu))
            torch.cuda.set_device(gpu_idx)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print("Training starts.")
    best_accur = train(args, log_name)
    print("Training finished.")
    print("Best accuracy:", best_accur)

