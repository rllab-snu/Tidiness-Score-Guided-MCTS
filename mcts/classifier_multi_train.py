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

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=4, args=None):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model
        if args.model=='resnet-18':
            resnet = resnet18
        elif args.model=='resnet-34':
            resnet = resnet34
        elif args.model=='resnet-50':
            resnet = resnet50

        if args.finetune:
            self.resnet = resnet(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            self.resnet = resnet(pretrained=False)

        # Remove the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Removing the original fully connected layer

        self.fc_class = nn.Linear(num_ftrs, num_classes)
        self.fc_score = nn.Linear(num_ftrs, 1)
        self.softmax = nn.Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        # Get class distribution
        logit = self.fc_class(x)
        class_distribution = self.softmax(logit)
        # Get the 1-dim score
        score = self.fc_score(x)
        #score = self.sigmoid(score)
        return score, logit, class_distribution

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
    dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'train'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, scene_index=True)
    us_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-seen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, scene_index=True)
    su_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-seen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, scene_index=True)
    uu_test_dataset = TabletopTemplateDataset(data_dir=os.path.join(args.data_dir, 'test-unseen_obj-unseen_template'), remove_bg=args.remove_bg, label_type=args.label_type, view=args.view, scene_index=True)
    print('len(train_dataset):', len(dataset))
    print('len(us_test_dataset):', len(us_test_dataset))
    print('len(su_test_dataset):', len(su_test_dataset))
    print('len(uu_test_dataset):', len(uu_test_dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    us_test_dataloader = DataLoader(us_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    su_test_dataloader = DataLoader(su_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    uu_test_dataloader = DataLoader(uu_test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    if args.finetune:
        model_name = 'finetune'
    else:
        # {view}_{remove_bg}_{label_type}_{loss}
        model_name = 'multi'
        model_name += '_nobg_' if args.remove_bg else '_bg_'
        model_name += args.label_type + '_'
        model_name += args.loss

    # model #
    print("Create a ResNet model.")
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    model = CustomResNet18(4, args)
    model.to(device)

    # loss function and optimizer #
    if args.loss=='mse':
        loss_score = nn.MSELoss()
    elif args.loss=='bce':
        loss_score = nn.BCELoss()  # binary cross entropy
    loss_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
 
    # Hold the best model
    best_accuracy = - np.inf   # init to negative infinity
    best_weights = None
 
    count_steps = 0
    for epoch in range(n_epoch):
        model.train()
        with tqdm(dataloader) as bar:
            bar.set_description(f"Epoch {epoch}")
            for X_batch, Y_batch, S_batch in bar:
                X_batch = preprocess(X_batch).to(device)
                Y_batch = Y_batch[:, 0].to(device)
                # forward pass
                y_pred, logit_pred, dist_pred = model(X_batch)
                B = y_pred.size(0)
                #y_pred = y_pred[torch.arange(B), S_batch]
                loss = loss_score(y_pred, Y_batch) + loss_class(logit_pred, S_batch.to(device))
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                indices = torch.logical_or(Y_batch==0, Y_batch==1)
                tidyup_acc = (y_pred[:,0].round() == Y_batch)[indices].float().mean()
                _, predicted = torch.max(dist_pred, 1)
                class_acc = (predicted.cpu() == S_batch).float().mean()

                bar.set_postfix(
                    loss=float(loss),
                    tidyup_acc=float(tidyup_acc),
                    class_acc=float(class_acc)
                )
                if not args.wandb_off:
                    step_log = {'train loss': float(loss), 
                                'tidyup accuracy': float(tidyup_acc), 
                                'class accuracy': float(class_acc)}
                    wandb.log(step_log)
                count_steps += 1
                break

        # evaluate accuracy at end of each epoch
        model.eval()
        tidyup_accuracies = []
        class_accuracies = []
        for test_dataloader in [us_test_dataloader, su_test_dataloader, uu_test_dataloader]:
            matchings = []
            class_matchings = []
            for X_val, Y_val, S_val in test_dataloader:
                X_val = preprocess(X_val).to(device)
                Y_val = Y_val[:, 0].to(device)
                y_pred, logit_pred, dist_pred = model(X_val)
                #y_pred = model(X_val)
                B = y_pred.size(0)
                #y_pred = y_pred[torch.arange(B), S_val]

                indices = torch.logical_or(Y_val==0, Y_val==1)
                matching = (y_pred[:,0].round()==Y_val)[indices].float().detach().cpu().numpy()
                matchings.append(matching)
                _, predicted = torch.max(dist_pred, 1)
                class_matching = (predicted.cpu()==S_val).float().detach().cpu().numpy()
                class_matchings.append(class_matching)
            matchings = np.concatenate(matchings, axis=0)
            class_matchings = np.concatenate(class_matchings, axis=0)
            tidyup_accuracy = np.mean(matchings)
            tidyup_accuracies.append(tidyup_accuracy)
            class_accuracy = np.mean(class_matchings)
            class_accuracies.append(class_accuracy)
        print("US validation accuracy:", tidyup_accuracies[0], class_accuracies[0])
        print("SU validation accuracy:", tidyup_accuracies[1], class_accuracies[1])
        print("UU validation accuracy:", tidyup_accuracies[2], class_accuracies[2])
        print("mean validation accuracy:", np.mean(tidyup_accuracies), np.mean(class_accuracies))
        if not args.wandb_off:
            step_log = {
                    'us valid tidyup accuracy': float(tidyup_accuracies[0]),
                    'su valid tidyup accuracy': float(tidyup_accuracies[1]),
                    'uu valid tidyup accuracy': float(tidyup_accuracies[2]),
                    'mean valid tidyup accuracy': float(np.mean(tidyup_accuracies)),
                    'us valid class accuracy': float(class_accuracies[0]),
                    'su valid class accuracy': float(class_accuracies[1]),
                    'uu valid class accuracy': float(class_accuracies[2]),
                    'mean valid class accuracy': float(np.mean(class_accuracies)),
                    }
            wandb.log(step_log)
        if np.mean(tidyup_accuracies) > best_accuracy:
            best_accuracy = np.mean(tidyup_accuracies)
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
    parser.add_argument("--label_type", type=str, default='linspace') # linspace / binary
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

