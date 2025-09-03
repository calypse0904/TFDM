import numpy as np
from PIL import Image
import imageio
import os


def writeRecords(list, path, imgName):
    name = imgName.split(".")[0];
    images = []
    for img in list:
        image = Image.fromarray(img)
        images.append(image)
    tmp_images = []
    tmp_folder = path + "/png"
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    for i, image in enumerate(images):
        temp_file = os.path.join(tmp_folder, name + f"_{i}.png")
        image.save(temp_file)
        tmp_images.append(temp_file)

    imageio.mimsave(os.path.join(path, name + ".gif"), [imageio.imread(img) for img in tmp_images],
                    fps=2)  # fps参数控制动画的帧率
