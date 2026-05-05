### ReciPI: Auto Recipe Generation​

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi%203%20B%20v1.2-C51A4A.svg)](https://www.raspberrypi.org/)

## Project Overview
This project uses edge computing to identify food ingredients in real-time and recommend recipes based on what you have available. Running entirely on a Raspberry Pi 3, the system uses a connected Pi Camera to capture images of ingredients, runs inference locally using a lightweight, quantized and prunned ResNet-18 model, and uses a recipe database to suggest a recipe using detected ingredient.

## Implementation Details
    Platform: Raspberry Pi 3 Model B v1.2 
    Peripherals: Raspberry Pi Camera Module V2

## Setup Instructions
1. Clone the repo
2. Connect the PiCamV2 to the Pi
3. Download the following files to the Pi:
    - hw_scripts/reciPI_main.py
    - JSON/id2label.json
    - data/reduced_recipe_cache.pkl
    - training_scripts/produce_net_fullmodel.pth
4. In the on-device command line, run ```python3 reciPI_main.py```
## Architecture

```text
+-----------------+      +-----------------+      +-----------------+      +-----------------+
|                 |      |                 |      |                 |      |                 |
|  Pi Camera V2   |----->|    ResNet-18    |----->|  Recipe Lookup  |----->|    Recipe       |
| (Image Capture) |      |                 |      |      Table      |      |  Recomendation  |
|                 |      |                 |      |                 |      |                 |
+-----------------+      +-----------------+      +-----------------+      +-----------------+
        |                         |                        |                        |
  Hardware Input            Edge Inference           Python Code             Display Output


