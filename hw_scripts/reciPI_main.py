import json
import time
import argparse

import torch
from torch import nn
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from picamera2 import Picamera2, Preview

#Load mean/std from dataset 
def load_norm_stats(path: str):

    with open(path, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    return mean, std

#From picamera2 docs
def capture_frame_rgb(quantity):
    images = []
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration())
    cam.start_preview(Preview.QT)
    cam.start()
    for _ in range(quantity):
        time.sleep(5)
        frame = cam.capture_array()
        print("Picture Captured!")
        images.append(Image.fromarray(frame).convert("RGB"))
    return images

def main():
    # Model Files
    mod = "../training_scripts/produce_net_fullmodel.pth"
    labels = "../JSON/id2label_produce.json"
    norm = "../JSON/norm_stats_produce_newdata.json"
    topk = 3 #number of predictions to display per ingredient

    # Welcome message
    print("------------------------------------------------------------------------")
    print("                       Welcome to ReciPI!")
    print("------------------------------------------------------------------------")
    print("\n")

    # Take pictures of ingredients with PI Camera
    images = [] #array to temporarily store pictures
    response = input("How many ingredients would you like to capture? ")
    capture_mode = True
    while(capture_mode):
        if not(response.isdigit()):
            print("Please input a valid number")
        elif(int(response == 0)):
            print("Must have at least one ingredient")
        else:
            images = capture_frame_rgb(quantity=int(response)) #take the requested number of pictures with the pi cam
            capture_mode = False

    # Load model
        with open(labels, "r") as f: #load label mapping
            id2label = json.load(f)
        
        mean, std = load_norm_stats(norm) #load normalization stats

    device = torch.device("cpu") #run on rpi3 cpu

    model = torch.load(mod, map_location=device, weights_only=False) #load model
    model.eval()

    # Prep images for inference
    transform = transforms.Compose([ #Define preprocessing for image before CNN
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    ])

    transformed_images = []
    for img in images:
        transformed_images.append(transform(img)) #apply image transform

    batch = torch.stack(transformed_images) #form input tensor

    # Run inference
    ingredients = [] #store predicted ingredients

    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch) #send input to model
        probs = torch.softmax(logits, dim=1)
        
        for i in range(0, len(batch)):
            topk = min(topk, probs[i].numel())
            vals, idxs = torch.topk(probs[i], k=topk) #Top k ingredients

            print("Choose the Appropriate Prediction for Ingredient", i) #present user with prediction
            for v, i in zip(vals.tolist(), idxs.tolist()):
                #JSON keys may be strings
                name = id2label.get(str(i), id2label.get(i, f"class_{i}"))
                print(f"  {name}: {v:.3f}")

                choice = input("Is this correct (y/n)") #prompt user to check prediction

                if(choice == 'y'): #correct prediciton
                    ingredients.append(name)
                    break
            
if __name__ == "__main__":
    main()
        
