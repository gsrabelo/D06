from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import pathlib
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DetectaObjetos:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detectar_objetos(self, imagem_pil, conf=0.25):
        resultados = self.model.predict(source=imagem_pil, conf=conf, iou=0.45, imgsz=640, verbose=False,)
        classes_detectadas = []
        for result in resultados:
            print(60*"-")
            print(f"Shape:  {result.orig_shape}")
            print(f"Objetos detectados: {len(result.boxes)}")
            print(60*"-")
            # os resultados são retângulos (bounding box) das detecções
            for retang in result.boxes:
                cls_id   = int(retang.cls[0])
                cls_nome = result.names[cls_id]
                classes_detectadas.append(cls_nome)
                conf_val = float(retang.conf[0])
                x1, y1, x2, y2 = retang.xyxy[0].tolist()
                cx, cy, w, h = retang.xywhn[0].tolist()
                print(f"  [{cls_id:2d}] {cls_nome:<15}  conf={conf_val:.2%}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        num_deteccoes = len(classes_detectadas)
        num_classes_detectadas = len(set(classes_detectadas))
        return resultados, num_deteccoes, num_classes_detectadas


    def visualizar_resultado(self, resultados, num_det, imagem_pil, titulo="Detecção", saida="output/deteccao_yolo.png"):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        img = np.array(imagem_pil)
        ax.imshow(img)
        cores = plt.cm.Set1(np.linspace(0, 1, max(num_det, 1)))
        for result in resultados:
            for retang in result.boxes:
                cls_id   = int(retang.cls[0])
                cls_nome = result.names[cls_id]
                conf = float(retang.conf[0])
                x1, y1, x2, y2 = retang.xyxy[0].cpu().numpy().astype(int)
                color = cores[cls_id % len(cores)]
                # retangulo deteccao
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2.5, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                # rótulo com confiança
                ax.text(
                    x1, y1 - 6, f"{cls_nome} {conf:.2f}",
                    color="white", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
                )

# --------------------------------
# instanciar objetos do endpoint
# (Modelo Yolo)
# --------------------------------
detector = DetectaObjetos()