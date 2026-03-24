from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


img_pil = Image.open('./A01/images/mulher_faixa_pedestre_pb.jpg')
img_np = np.array(img_pil)
dados = f"Dimensões da imagem: {img_np.shape} (altura x largura x canais)\n"
dados += f"Modo da imagem: {img_pil.mode}\n"
dados += f"Formato da imagem: {img_pil.format}\n"

img_np[40:65, 70:95] = 255

fig = plt.figure(figsize=(6, 4))
imgplot = plt.imshow(img_np)
plt.show()

print(f"Imagem\n\n{dados}")

#display(img_pil)