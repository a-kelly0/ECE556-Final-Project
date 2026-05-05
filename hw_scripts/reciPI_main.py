import json
import time
import argparse
import psutil
import os

import torch
from torch import nn
from PIL import Image
import pickle
import time

import torchvision.transforms as transforms
import torchvision.models as models
from picamera2 import Picamera2, Preview

def load_norm_stats(path: str):
    '''
    Loads mean and standard deviation dataset statistics from json file
    '''
    with open(path, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    return mean, std

def capture_frame_rgb(quantity):
    '''
    Captures the provided number of pictures with the Raspberry PI camera.
    Console print after each picture is taken.
    '''
    images = []
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration())
    cam.start_preview(Preview.QT)
    cam.start()
    for _ in range(quantity):
        time.sleep(5)
        frame = cam.capture_array()
        print("\nPicture Captured!\n")
        images.append(Image.fromarray(frame).convert("RGB"))
    cam.close()
    return images

def get_recipe(ingredients:list):
    '''
    Given a list of ingredients, searches the recipe dictionary for the top recipe that uses the available ingredients
    Dictionary of valid recipes should be stored as reduced_recipe_cache.pkl in the current dir
    '''
    with open("../data/reduced_recipe_cache.pkl", "rb") as f:
        ds = pickle.load(f)

    ingredients = set(ing.strip().lower() for ing in ingredients)

    winner = None
    best_score = -1
    link = None

    #iterate through recipes, rank by how many of the available ingredients are used
    for row in ds:
        score = sum(1 for ing in row["NER"] if ing.strip().lower() in ingredients)

        if score > best_score:
            best_score = score
            winner = row["title"]
            link = row["link"]

    #if no recipes use this ingredient
    if(best_score == 0):
        print("Could not find recipe in the dataset for these ingredients")
        winner = None    
    return winner, link

def main():
    '''
    Main system loop for ReciPI.
    Captures ingredients, runs inference, and returns the provided recipe.
    '''
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
    print("\n")
    print("---------------------------Starting PI Camera---------------------------")
    capture_mode = True
    while(capture_mode):
        if not(response.isdigit()):
            print("Please input a valid number")
        elif(int(response == 0)):
            print("Must have at least one ingredient")
        else:
            images = capture_frame_rgb(quantity=int(response)) #take the requested number of pictures with the pi cam
            capture_mode = False
    print("---------------------------Processing Images---------------------------")

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
        #process = psutil.Process(os.getpid())
        #process.cpu_percent(interval=None)
        #t1 = time.time()
        logits = model(batch) #send input to model
        print("RAM Usage:", psutil.virtual_memory().total - psutil.virtual_memory().available)
        print("Total RAM", psutil.virtual_memory().total)
        #cpu_usage = process.cpu_percent(interval=None) #get active cpu usage
        #t2 = time.time()
        #print("inference latency:", t2-t1) #get inference latency
        #print("Inference CPU Usage", cpu_usage)

        probs = torch.softmax(logits, dim=1)
        
        for i in range(0, len(batch)):
            topk = min(topk, probs[i].numel())
            vals, idxs = torch.topk(probs[i], k=topk) #Top k ingredients

            print("Choose the Appropriate Prediction for Ingredient", i+1) #present user with prediction
            for v, i in zip(vals.tolist(), idxs.tolist()):
                #JSON keys may be strings
                name = id2label.get(str(i), id2label.get(i, f"class_{i}"))
                print(f"  {name}: {v:.3f}")

                choice = input("Is this correct (y/n)") #prompt user to check prediction

                if(choice == 'y'): #correct prediciton
                    ingredients.append(name)
                    break

    # Find matching recipe
    print("\nYour ingredients are:", ingredients, "\n")
    print("----------------------_-----Finding Recipe---------_------------------")
    t1 = time.time()
    rec, link = get_recipe(ingredients=ingredients)
    t2 = time.time()
    print("recipe search latency:", t2-t1)
    if rec != None:
        print("Recommended Recipe:", rec)
        print("Link:", link)
            
if __name__ == "__main__":
    main()
        
