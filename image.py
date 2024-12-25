import cv2
import numpy as np
import argparse
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from cp.onnx_model import face_parser


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image', default='ti.jpg')
    parse.add_argument('--result', default='ti.jpg')
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def make(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part != 17:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    # Make sure the dimensions of parsing match those of the image
    parsing_resized = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Ensure parsing_resized has the same number of channels as the image
    if parsing_resized.ndim == 2:
        parsing_resized = np.expand_dims(parsing_resized, axis=-1)  # Expand to (H, W, 1)

    # Expand parsing_resized to have the same 3 channels as the image
    parsing_resized = np.repeat(parsing_resized, 3, axis=-1)  # Expand to (H, W, 3)

    # Now both `image` and `parsing_resized` should have the same number of channels
    changed[parsing_resized != part] = image[parsing_resized != part]

    return changed
