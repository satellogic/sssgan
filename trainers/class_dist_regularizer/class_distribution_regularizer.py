import torch
import torch.nn as nn
from .class_dist_model import ClassDistModel
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
class SatelliteDistributionRegularizer(nn.Module):
    def __init__(self, dp=None, device_ids=None):
        super().__init__()
        img_size=256
        dim=2
        model = ClassDistModel(load=False, output_dim=dim)
        if dp is not None:
            model = dp(model, device_ids=device_ids)
        else:
            model = nn.DataParallel(model)
        checkpoint = torch.load('/home/emilioty/ws/SPADE/trainers/class_dist_regularizer/weights/only#vehicles.pt')#, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        del checkpoint
        self.model = model
        self.loss = nn.MSELoss()
        #for param in self.parameters():
        #        param.requires_grad = False

        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2359, 0.1949 , 0.1598), (0.1176, 0.0961, 0.0864))
        ])
        self.categories = np.loadtxt("/home/emilioty/ws/SPADE/trainers/class_dist_regularizer/classes.txt", dtype=str)
        
    def forward(self, real_vec, fake):
        return self.loss(real_vec.detach(), self.model(fake))

    def count_classes(self, path, name):
        img = Image.open(path)
        tensor = self.data_transforms(img)
        tensor = torch.unsqueeze(tensor, dim=0)
        tensor.to(self.device)
        pred = self.model(tensor).detach().cpu().numpy()[0]
        pred = [round(p) for p in pred]
        df = pd.DataFrame(np.array([pred]), columns = self.categories, index=[name])
        return df
        