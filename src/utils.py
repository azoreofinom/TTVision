import cv2
import numpy as np
import torch
from PIL import Image


def tensor2array(tensor):
    return tensor.detach().cpu().numpy()

# Converts the image from PIL to tensor of the right size
# Takes as input a list of pil_images
def preprocess(pil_imgs, size=(640, 320), device="cpu"):
    resized_tensors = []
    for img in pil_imgs:
        img_array = np.array(img)
        resized = cv2.resize(img_array, size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1))
        resized_tensors.append(tensor[None].to(device))
    return torch.cat(resized_tensors)


def postprocess(t_masks, sizes):
    num_masks = t_masks.shape[0]
    resized_masks = []
    for i in range(num_masks):
        mask = tensor2array(t_masks[i].squeeze())
        mask = 255 * mask
        resized_masks.append(mask)

    return resized_masks


def read_img(img_path):
    return Image.open(img_path).convert("RGB")
