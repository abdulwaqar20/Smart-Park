# Smart Parking System

This project detects empty and filled parking spots in images and videos using a YOLOv8 object detection model. It has two main parts: a training script and a web app to test the model.

## Files

- **train.py** — Trains the YOLOv8 model on your parking lot dataset.
- **Interface.py** — A Streamlit web app where you can upload an image or video and see detected parking spots marked in color (green = empty, red = filled).

## Requirements

Install these Python packages before running the project:

```
pip install ultralytics opencv-python streamlit numpy torch
```

## How Training Works (train.py)

- It checks if a GPU is available. If yes, it uses `cuda`; otherwise it uses `cpu`.
- It loads a fresh YOLOv8m model structure from `yolov8m.yaml`.
- It trains the model using a dataset described in `config.yaml`, for 20 epochs.

### Run training

```
python train.py
```

Make sure you have a `config.yaml` file in the same folder that points to your dataset (images and labels for empty/filled parking spots).

After training, YOLO will save the trained weights inside a folder like:

```
runs/detect/trainXX/weights/last.pt
```

## How the Interface Works (Interface.py)

- Loads the trained model from `runs/detect/train23/weights/last.pt`.
- Lets you upload an image or video file (`.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, `.mov`).
- Lets you set a confidence threshold using a slider (only detections above this score are shown).
- For images: runs detection once and shows the result with colored boxes.
- For videos: plays the video and processes it frame by frame, showing detection boxes live.
- Green box = empty spot, Red box = filled spot.

### Run the interface

```
streamlit run Interface.py
```

This will open a browser window where you can upload your file and test the model.

## Notes

- Before running `Interface.py`, make sure the model path (`runs/detect/train23/weights/last.pt`) matches the actual folder created after your training run. If your training created a different folder name (like `train1`, `train5`, etc.), update the path in the code.
- Video processing works better with shorter videos since each frame goes through the model one by one.