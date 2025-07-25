# Thesis Action Recognition Scripts

This repository contains a collection of small utilities used during my thesis work on skeleton‑based human action recognition. Most scripts are written in Python and focus on converting raw skeleton data into images, splitting datasets and training convolutional neural networks.

## Repository layout

- `mp4/` – video to skeleton pipeline using MediaPipe. See [`mp4/README.md`](mp4/README.md) for details.
- `image transformation/` – convert skeleton CSV files to FFT or pseudo‑RGB images.
- `split/` – helper scripts to organize datasets into train/val/test folders.
- `train/` – training scripts for models such as ResNet50 and MobileNetV2.
- `visualization/` – utilities for plotting skeleton sequences or generated datasets.
- `create_labels/` and other root‑level scripts – miscellaneous helpers (counting files, generating label CSVs, etc.).

All scripts contain hard coded file paths that were specific to the original experiments. Before running them, adjust the paths to match your own dataset locations.

## Dependencies

The code relies mainly on:

- Python 3
- TensorFlow and Keras
- NumPy, SciPy, Matplotlib
- OpenCV and MediaPipe (for the video pipeline)

Install the required packages with pip for your environment, e.g.:

```bash
pip install tensorflow opencv-python mediapipe numpy scipy matplotlib
```

## Usage

1. Extract skeletons from videos using scripts in `mp4/`.
2. Convert skeleton CSV files to images with the scripts in `image transformation/`.
3. Split the generated images into train/val/test folders using the tools under `split/`.
4. Train a CNN of your choice with the code in `train/`.

The individual scripts show example settings for the NTU and PKU datasets. Adapt them as needed for your own experiments.

