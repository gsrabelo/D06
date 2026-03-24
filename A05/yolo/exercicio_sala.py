# Visao Computacional e IA Generativa

# 1.2. Representação digital de imagens (PIL e OpenCV)
# Exercícios práticos (criar ambiente / comandos)
# > powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/inst
# > .\venv-gen-ai\Scripts\activate
# > uv pip install cmake
# > uv pip install wheel
# > uv pip install torch torchvision scikit-learn scikit-image matplotlib timm
# > uv pip install fastapi gradio pydantic ipython jupyter dlib
# > uv pip install opencv-python opencv-contrib-python

----------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Ultralytics / YOLOv8
from ultralytics import YOLO
yolo_seg = YOLO("yolov8n-seg.pt")

def segmentar_pessoas(img_rgb, yolo_seg):
    h, w = img_rgb.shape[:2]
    results = yolo_seg(img_rgb)
    r = results[0]
    person_class_id = 0
    final_mask = np.zeros((h, w), dtype=np.uint8)
    if r.masks is not None and r.boxes is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy()  # shape: [N, Hm, Wm]

        for cls_id, mask in zip(classes, masks):
            if cls_id == person_class_id:
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                final_mask = np.maximum(final_mask, mask_resized)

    result_rgb = np.zeros_like(img_rgb)
    result_rgb[final_mask > 0] = img_rgb[final_mask > 0]
    return result_rgb

img_bgr = cv2.imread("data/t1.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
imagem_segmentada = segmentar_pessoas(img_rgb, yolo_seg)
display(Image.fromarray(imagem_segmentada))

