#ReciPI: Auto Recipe Generation​

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Platform: Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi%203-C51A4A.svg)](https://www.raspberrypi.org/)

## Project Overview
This project uses edge computing to identify food ingredients in real-time and recommend recipes based on what you have available. Running entirely on a Raspberry Pi 3, the system uses a connected Pi Camera to capture images of ingredients, runs inference locally using a lightweight, quantized ResNet-18 model, and queries a recipe database to suggest meals.

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
