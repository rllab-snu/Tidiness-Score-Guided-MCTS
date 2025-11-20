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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50


preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'train'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, target_scene=args.target_scene)
    print('len(train_dataset):', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)

    # model #
    print("Create a ResNet model.")
    if args.model=='resnet-18':
        resnet = resnet18
    elif args.model=='resnet-34':
        resnet = resnet34
    elif args.model=='resnet-50':
        resnet = resnet50
    if args.finetune:
        model = resnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model_name = 'finetune'
    else:
        model = resnet(pretrained=False)
        # {view}_{remove_bg}_{label_type}_{loss}
        model_name = args.target_scene #view.replace('_', '')
        model_name += '_nobg_' if args.remove_bg else '_bg_'
        model_name += args.label_type + '_'
        model_name += args.loss
    # replace the last fc layer #
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, 1),
        #nn.Sigmoid()
    )

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
 
    count_steps = 0
    for epoch in range(n_epoch):
        model.train()
        with tqdm(dataloader) as bar:
            bar.set_description(f"Epoch {epoch}")
            for X_batch, Y_batch in bar:
                X_batch = preprocess(X_batch).to(device)
                Y_batch = Y_batch[:, 0].to(device)
                # forward pass
                y_pred = model(X_batch)[:, 0]
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

        if False:
            # evaluate accuracy at end of each epoch
            model.eval()
            accuracies = []
            for test_dataloader in [us_test_dataloader, su_test_dataloader, uu_test_dataloader]:
                matchings = []
                for X_val, Y_val in test_dataloader:
                    X_val = preprocess(X_val).to(device)
                    Y_val = Y_val[:, 0].to(device)
                    y_pred = model(X_val)[:, 0]
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
        if float(acc) > best_accuracy:
            best_accuracy = float(acc)
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
    parser.add_argument("--data_dir", type=str, default='/disk1/hogun/TableTidyingUp/dataset_shape')
    parser.add_argument("--remove_bg", action="store_true") # default: False
    parser.add_argument("--label_type", type=str, default='binary') # linspace / binary
    parser.add_argument("--view", type=str, default='top') # top / front_top
    parser.add_argument("--target_scene", type=str, default='line') # line / circle / B / C
    # etc
    parser.add_argument("--model", type=str, default='resnet-18')
    parser.add_argument('--wandb-off', action='store_true')
    args = parser.parse_args()
    args.remove_bg = True

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

