'''
modified Phil Wang's code
url: https://github.com/lucidrains/byol-pytorch
'''
import os

import torch
from torch import nn
from torch.optim import Adam
from torchvision import models
from torchvision import transforms as T
from torch.utils.data import DataLoader

import sys

from tqdm import tqdm

import wandb
import random
import argparse
from byol_pytorch import BYOL

import config
from self_datasets import MotionDataset

from utils import Identity


def BYOL_train(params,):
    params['representation'] = 1

    if (params['wandb']):
        wandb.init(project = 'imitation_byol'+"_"+str(params['version']))

    customAug = T.Compose([T.RandomResizedCrop(params['img_size'], scale=(0.6,1.0)),
                            T.RandomApply(torch.nn.ModuleList([T.ColorJitter(.8,.8,.8,.2)]), p=.3),
                            T.RandomGrayscale(p=0.2),
                            T.RandomApply(torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2),
                            T.Normalize(
                            mean=torch.tensor([0.485, 0.456, 0.406]),
                            std=torch.tensor([0.229, 0.224, 0.225]))])

    device = params['device']
    print("use device : "+ device)
    if params["model_weight"] == "":
        encoder = models.resnet50(pretrained=True).to(device)
    else:
        encoder = models.resnet50().to(device)
        encoder.load_state_dict(torch.load(params["model_weight"], map_location=torch.device('cpu')))
        encoder.fc = Identity()
        print("load weight: " + params["model_weight"])

    if (params['wandb']):
        wandb.watch(encoder)

    learner = BYOL(
        encoder,
        image_size=params['img_size'],
        hidden_layer='avgpool',
        augment_fn=customAug
    )

    img_data = MotionDataset(params, None)
    data_loader = DataLoader(img_data, batch_size=params['batch_size'], shuffle=True, pin_memory=True, num_workers=8)
    optimizer = Adam(learner.parameters(), lr=3e-4, weight_decay=1.5e-06)

    for epoch in range(params['epochs']):
        epoch_loss = 0
        loop = tqdm(data_loader, leave=True, desc="Training")
        for i, data in enumerate(loop, 0):
            loss = learner(data.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()

            epoch_loss += loss.item() * data.shape[0]

            loop.set_postfix(loss=loss.item(), epoch_loss=epoch_loss / len(img_data))

        # print(epoch_loss / len(img_data))
        if (params['wandb']):
            wandb.log({"epoch_loss": epoch_loss / len(img_data)})

        if (epoch % 10 == 0):
            if not os.path.exists(params['save_dir']):
                os.mkdir(params['save_dir'])
            torch.save(encoder.state_dict(),
                       params['save_dir'] + 'BYOL_' + str(epoch) + '_' + str(params['version']) + '.pt')


if __name__ == '__main__':
    BYOL_train(config.train_model)



