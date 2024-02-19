import pickle

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

import config
from self_datasets import MotionDataset
from servo_utils import uart, servoToAngle

from utils import Identity





class initial_stage:
    def __init__(self,params):
        self.preprocess = T.Compose([T.ToTensor(),
                                     T.Resize((224, 224)),
                                     T.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
        self.nn_counter = 0
        self.softmax = torch.nn.Softmax(dim=0)
        params['representation'] = 0
        self.params = params
        if params["model_weight"] == "":
            self.model = models.resnet50(pretrained=True)
            self.model.fc = Identity()
        else:
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(params["model_weight"], map_location=torch.device('cpu')))
            self.model.fc = Identity()
            print("load weight: " + params["model_weight"])

        self.cap = cv2.VideoCapture(0)

    def dist_metric(self, x, y):
        return (torch.norm(x - y).item())

    def calculate_action(self, dist_list, k):
        # 这个action的长度是对应机械臂关节的长度
        action = torch.tensor([0.0, 0.0, 0.0,0.0,0.0])
        top_k_weights = torch.zeros((k,))
        for i in range(k):
            top_k_weights[i] = dist_list[i][0]


        top_k_weights = self.softmax(-1 * top_k_weights)
        for i in range(k):
            action = torch.add(top_k_weights[i] * dist_list[i][1], action)

        self.nn_counter += 1

        return action

    def calculate_nearest_neighbors(self,img_tensor, dataset, k):
        dist_list = []

        for dataset_index in range(len(dataset)):

            dataset_embedding, dataset_action, dataset_path = dataset[dataset_index]
            distance = self.dist_metric(img_tensor, dataset_embedding)
            dist_list.append((distance, dataset_action, dataset_path))

        dist_list = sorted(dist_list, key = lambda tup: tup[0])
        pred_action = self.calculate_action(dist_list, k)
        return pred_action

    def start_execution(self):
        k = 5
        img_count = 0
        dataset = MotionDataset(self.params, self.model)
        while self.cap.isOpened():
            ret, image = self.cap.read()
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            img_tensor = self.preprocess(im)
            im.close()

            embedding = self.model(img_tensor.reshape(1, 3, 224, 224))[0]

            action_tensor = self.calculate_nearest_neighbors(embedding, dataset, k).tolist()[0]

            print(action_tensor)

            uart.open()
            for i in range(len(action_tensor)):
                # servoToAngle(i+1, int(action_tensor[i]), 10)
                print(f"关节{i+1} 的角度{int(action_tensor[i])}")

            uart.close()

            img_count += 1


if __name__ == '__main__':
    i = initial_stage(config.train_model)
    i.start_execution()