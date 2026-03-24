from fastapi import FastAPI
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/teste")
def read_root():
    return {"mensagem": "Teste simples do FastAPI!"}


@app.get("/dados/{img_path}")
def read_item(img_path: str):
    img_pil = Image.open('./A01/images/hal9000.png')
    img_array = np.array(img_pil)
    dados = f"Dimensões da imagem: {img_array.shape} (altura x largura x canais)\n"
    dados += f"Modo da imagem: {img_pil.mode}\n"
    dados += f"Formato da imagem: {img_pil.format}\n"

    fig = plt.figure(figsize=(6, 4))
    plt.imshow(img_array)
    
    return {"mensagem": f"Imagem {img_path}\n\n{dados}"}