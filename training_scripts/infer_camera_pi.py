import json
import time
import argparse

import torch
from torch import nn
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from picamera2 import Picamera2

#Recreate exact architecture trained
def build_model(num_classes: int) -> torch.nn.Module:
    #For inference, weights can be None because we load fine-tuned state_dict anyway
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

#Load mean/std from dataset 
def load_norm_stats(path: str):

    with open(path, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    return mean, std

#From picamera2 docs
def capture_frame_rgb() -> Image.Image:
    cam = Picamera2()
    #Force RGB format to avoid BGR
    cam.configure(cam.create_still_configuration(main={"format": "RGB888"}))
    cam.start()
    time.sleep(1.0)
    frame = cam.capture_array()
    cam.stop()
    return Image.fromarray(frame).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="food_net.pth", help="Path to saved model state_dict")
    ap.add_argument("--labels", default="JSON/id2label.json", help="Path to id2label.json")
    ap.add_argument("--norm", default="JSON/norm_stats.json", help="Path to normalization stats json")
    ap.add_argument("--image", default=None, help="Optional: run inference on an image file instead of camera")
    ap.add_argument("--topk", type=int, default=5, help="How many top predictions to print")
    args = ap.parse_args()

    #Load label mapping
    with open(args.labels, "r") as f:
        id2label = json.load(f)

    #Load normalization stats
    mean, std = load_norm_stats(args.norm)

    #Define preprocessing for image before CNN
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    #Run on rpi3 cpu
    device = torch.device("cpu")

    model = build_model(num_classes=len(id2label)).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    #Get image
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
    else:
        img = capture_frame_rgb()

    #Add batch dimension
    x = transform(img).unsqueeze(0).to(device)

    #No gradients to save memory
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

        topk = min(args.topk, probs.numel())
        vals, idxs = torch.topk(probs, k=topk) #Top 5 ingredients

    print("Top predictions:")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        #JSON keys may be strings
        name = id2label.get(str(i), id2label.get(i, f"class_{i}"))
        print(f"  {name}: {v:.3f}")


if __name__ == "__main__":
    main()