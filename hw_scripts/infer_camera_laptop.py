import json
import argparse
from pathlib import Path

import torch
from torch import nn
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models


# Recreate exact architecture (used only if checkpoint is a state_dict)
def build_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_norm_stats(path: str):
    with open(path, "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std"]
    return mean, std


def load_model_flexible(weights_path: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """
    Loads a model from a .pth that might be:
      - a full torch.nn.Module (your case: ResNet object)
      - a state_dict (OrderedDict)
      - a checkpoint dict containing state_dict under common keys
    """
    obj = torch.load(weights_path, map_location=device, weights_only=False)

    # Case 1: full model object saved
    if isinstance(obj, torch.nn.Module):
        model = obj

        # Safety check: make sure output dimension matches labels
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            out_features = model.fc.out_features
            if out_features != num_classes:
                raise ValueError(
                    f"Model fc out_features ({out_features}) != num_classes from labels ({num_classes}). "
                    "Your id2label.json may not match the trained model."
                )
        return model.to(device)

    # Case 2: dict checkpoint or state_dict
    state_dict = None

    if isinstance(obj, dict):
        # common checkpoint patterns
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in obj:
                if key == "model" and hasattr(obj["model"], "state_dict"):
                    # model object stored inside dict
                    model = obj["model"].to(device)
                    return model
                state_dict = obj[key]
                break
        if state_dict is None:
            # maybe the dict itself is the state_dict
            state_dict = obj
    else:
        # maybe it's already an OrderedDict state_dict
        state_dict = obj

    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    return model


def main():
    HERE = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=str((HERE / ".." / "training_scripts" / "produce_net.pth").resolve()),
        help="Path to saved model (.pth). Can be full model or state_dict checkpoint.",
    )
    ap.add_argument(
        "--labels",
        default=str((HERE / ".." / "JSON" / "id2label_produce.json").resolve()),
        help="Path to id2label.json",
    )
    ap.add_argument(
        "--norm",
        default=str((HERE / ".." / "JSON" / "norm_stats_produce.json").resolve()),
        help="Path to normalization stats json",
    )
    ap.add_argument("--image", required=True, help="Run inference on an image file")
    ap.add_argument("--topk", type=int, default=5, help="How many top predictions to print")
    args = ap.parse_args()

    # Load label mapping
    with open(args.labels, "r") as f:
        id2label = json.load(f)

    # num_classes should match model output size
    num_classes = len(id2label)

    # Load normalization stats
    mean, std = load_norm_stats(args.norm)

    # Define preprocessing for image before CNN
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # CPU inference (safe for Pi too)
    device = torch.device("cpu")

    # Load model (handles "full model saved" case)
    model = load_model_flexible(args.weights, device=device, num_classes=num_classes)
    model.eval()

    # Get image
    img = Image.open(args.image).convert("RGB")

    # Add batch dimension
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

        topk = min(args.topk, probs.numel())
        vals, idxs = torch.topk(probs, k=topk)

    print("Top predictions:")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        # JSON keys may be strings
        name = id2label.get(str(i), id2label.get(i, f"class_{i}"))
        print(f"  {name}: {v:.3f}")


if __name__ == "__main__":
    main()
######################
# Run inference script
######################

# Switch to hw_scripts folder and run this command in the terminal:

#python infer_camera_laptop.py --image ../Captured_Ingredients/broccoli.jpg