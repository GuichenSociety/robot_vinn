import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import glob
from tqdm import tqdm
import json
from PIL import Image, ImageFile
import random
from collections import defaultdict

import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MotionDataset(Dataset):

    def __init__(self, params, encoder, partial=None):
        self.params = params
        self.encoder = encoder

        self.img_tensors = []
        self.representations = []
        self.action = []

        self.paths = []
        self.path_dict = defaultdict(list)
        self.frame_index = defaultdict(int)

        self.preprocess = T.Compose([T.ToTensor(),
                                    T.Resize((self.params['img_size'],self.params['img_size'])),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                     ])

        self.extract_data(partial)
        print("")

    def extract_data(self, factor=None):
        index = 0
        runs = glob.glob(self.params['folder']+'/*')
        random.shuffle(runs)
        total = len(runs)
        if factor is not None:
            total = int(total * factor)

        for run_index in tqdm(range(total),desc="Loading datasets..."):
            run = runs[run_index]
            action = glob.glob(run+'/*.json')

            for ac in action:
                action_file = open(ac, 'r')
                action_dict = json.load(action_file)

                img = Image.open(run +'/' + action_dict['img'])
                img = img.convert("RGB")
                img_tensor = self.preprocess(img)
                img.close()

                if(self.params['representation'] == 1):
                    self.img_tensors.append(img_tensor.detach())

                else:
                    c,w,h= img_tensor.shape
                    represnetation = self.encoder(img_tensor.reshape(1,c,w,h))[0]
                    self.representations.append(represnetation.detach())
                    self.action.append(torch.tensor([action_dict['action']]))
                    self.paths.append(runs[run_index]+'/'+action_dict['img'])
                    self.path_dict[runs[run_index]].append(action_dict['img'])
                    self.frame_index[runs[run_index] + '/' + action_dict['img']] = index
                index += 1

    def get_subset(self, factor):
        total = int(factor * len(self.path_dict))
        keys = random.sample(list(self.path_dict.keys()), total)
        items = []
        for k in keys:
            for v in self.path_dict[k]:
                items.append(self.__getitem__(self.frame_index[k + '/' +  v]))
        return items


    def __len__(self):
        return(max(len(self.img_tensors),len(self.representations)))

    def __getitem__(self, index):
        if(self.params['representation'] == 1):
                return(self.img_tensors[index])
        else:
            return((self.representations[index],self.action[index], self.paths[index]))

if __name__ == '__main__':
    b = config.train_model
    b['representation'] = 1
    a = MotionDataset(b,None)