import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image



def tensor2array(tensor):
    return tensor.detach().cpu().numpy()





# Converts the image from PIL to tensor of the right size
# Takes as input a list of pil_images
def preprocess(pil_imgs, size=(640, 320), device="cuda:0"):
    transform = A.Compose([A.Resize(height=size[1], width=size[0]), ToTensorV2()])
    resized_tensors = []
    for img in pil_imgs:
        img_array = np.array(img)
        transformed = transform(image=img_array)
        resized_tensors.append(transformed["image"][None].to(device))
    return torch.cat(resized_tensors)


def postprocess(t_masks, sizes):
    num_masks = t_masks.shape[0]
    resized_masks = []
    # print(f"post size:{sizes}")
    for i in range(num_masks):
        mask = tensor2array(t_masks[i].squeeze())
        mask = 255 * mask

        # resized_mask = cv2.resize(
        #     mask, (sizes[i][1], sizes[i][0]), interpolation=cv2.INTER_NEAREST
        # )
        # print(resized_mask.shape)
        resized_masks.append(mask)

    return resized_masks


def read_img(img_path):
    return Image.open(img_path).convert("RGB")
