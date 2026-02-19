#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import torchvision

import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#load datasets
ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train[:5000]")
ds = ds.train_test_split(test_size=0.2) #split data into training data and validation data
print(ds["train"][0])

#create dictionary that maps label name to integer
ingredients = sorted(set(ds["train"]["ingredient"]))
label2id = {name: str(i) for i, name in enumerate(ingredients)}
id2label = {str(i): name for i, name in enumerate(ingredients)}

print(id2label["0"])

#process image into tensor
transform = transforms.Compose([
    transforms.Resize((128,128)),          # Resize images to 128x128 (this number can be changed for different resolutions)
    transforms.ToTensor(),                 # Convert PIL image to tensor
    transforms.Normalize([0.5,0.5,0.5],    # Normalize
                         [0.5,0.5,0.5]) #TODO normalize to mean of dataset
])

def transform_fn(example):
    image = example["image"].convert("RGB") #make all images RGB to avoid initial channel mismatches
    example["image"] = transform(image)
    return example

ds["train"] = ds["train"].map(transform_fn)
ds["test"]  = ds["test"].map(transform_fn)

#create dataloaders
batch_size = 32

train_loader = DataLoader(dataset=ds["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=False)

#Simple cnn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 64 -> 32
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32, 128),  # 32 channels, 32x32 feature map
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    


