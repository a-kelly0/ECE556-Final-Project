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

#Hyperparameters
input_size = 128*128
learning_rate = 0.001
num_epochs = 10
batch_size = 32

#load datasets
ds = load_dataset("Scuccorese/food-ingredients-dataset", split="train[:500]")
ds = ds.train_test_split(test_size=0.2) #split data into training data and validation data
print(ds["train"][0])

#create dictionary that maps label name to integer
ingredients = sorted(set(ds["train"]["ingredient"]))
label2id = {name: str(i) for i, name in enumerate(ingredients)}
id2label = {str(i): name for i, name in enumerate(ingredients)}

num_classes = len(ingredients)

print(id2label["0"])

#process image into tensor
transform = transforms.Compose([
    transforms.Resize((128,128)),          # Resize images to 128x128 (this number can be changed for different resolutions)
    transforms.ToTensor(),                 # Convert PIL image to tensor
    transforms.Normalize([0.5,0.5,0.5],    # Normalize
                         [0.5,0.5,0.5]) #TODO normalize to mean of dataset
])

def transform_fn(data):
    image = data["image"].convert("RGB") #make all images RGB to avoid initial channel mismatches
    data["image"] = transform(image)
    data["label"] = int(label2id[data["ingredient"]])
    return data

ds["train"] = ds["train"].map(transform_fn)
ds["test"]  = ds["test"].map(transform_fn)

#convert to pytorch tensor
ds["train"] = ds["train"].with_format("torch", columns=["image", "label"])
ds["test"] = ds["test"].with_format("torch", columns=["image", "label"])

#create dataloaders

train_loader = DataLoader(dataset=ds["train"], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=ds["test"], batch_size=batch_size, shuffle=False)

# #display some random training images
# def imshow(img):
#     plt.imshow(np.transpose(img.numpy()), (1, 2, 0))
#     plt.show()

# dataiter = iter(train_loader)
# images, labels, = next(dataiter)

# imshow(torchvision.utils.make_grid(images))


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
    
model = SimpleCNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#only useful if using gpu, add back support for later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Loop through dataset and update the model's weights
for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#Find the accuracy of the network
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for data in val_loader:
        images = data["image"]
        labels = data["label"]
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 500 test images: {100 * correct // total} %')