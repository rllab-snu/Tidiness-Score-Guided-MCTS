import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import TabletopTemplateDataset
from torchvision.models import resnet18

bs = 256

data_paths = []
data_paths.append('/disk1/hogun/TableTidyingUp/dataset_template/test-unseen_obj-unseen_template')
data_paths.append('/disk1/hogun/TableTidyingUp/dataset_template/test-seen_obj-unseen_template')
data_paths.append('/disk1/hogun/TableTidyingUp/dataset_template/test-unseen_obj-seen_template')
#data_paths.append('/disk1/hogun/TableTidyingUp/dataset_template/train')

def scoring(model, dataloader):
    scores = {}
    for s in [0, 0.25, 0.5, 0.75, 1]:
        scores[s] = []

    for X_val, Y_val in dataloader:
        x_raw = X_val.cpu().numpy().transpose([0, 2, 3, 1])
        X_val = preprocess(X_val).to(device)
        y_val = Y_val[:, 0].to(device)
        y_pred = model(X_val)[:, 0].detach().cpu().numpy()

        for s in [0, 0.25, 0.5, 0.75, 1]:
            scores[s] += y_pred[Y_val[:, 0]==s].tolist()
    return scores

resnet = resnet18
model_path = 'data/classification-best/top_nobg_linspace_mse-best.pth'
device = "cuda:0"

model = resnet(pretrained=False)
fc_in_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(fc_in_features, 1),
    #nn.Sigmoid()
)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


th = 0.85
scores_all = []
labels_all = []
for di, data_path in enumerate(data_paths):
    dataset = TabletopTemplateDataset(data_dir=data_path,
                remove_bg=True, label_type='linspace', view='top')
    if di>0: dataset.fsize = 200
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=1)

    scores_raw = scoring(model, dataloader)
    scores = np.concatenate([scores_raw[s] for s in [0, 0.25, 0.5, 0.75, 1]])
    labels = np.concatenate([[s]*len(scores_raw[s]) for s in [0, 0.25, 0.5, 0.75, 1]])
    scores_all.append(scores)
    labels_all.append(labels)

scores = np.concatenate(scores_all)
labels = np.concatenate(labels_all)
np.save('scores_val.npy', scores)
np.save('labels_val.npy', labels)

#recall = np.array(np.array(scores[1])>th).sum() / len(scores[1])
accuracy = ((scores[labels==1]>th).sum() + (scores[labels==0]<th).sum()) / \
        (len(scores[labels==1]) + len(scores[labels==0]))
recall = np.array(np.array(scores[labels==1])>th).sum() / len(scores[labels==1])
precision = (labels[scores > th]==1).sum() / (scores > th).sum()
f1score = 2 * (recall * precision) / (recall + precision)
print('accuracy:', accuracy)
print('recall:', recall)
print('precision:', precision)
print('f1-score:', f1score)



