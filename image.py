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

if __name__ == '__main__':

    args = parse_args()

    table = {
        'face' : 1,
        'left_brow' : 2,
        'right_brow' : 3,
        'left_eye' : 4,
        'right_eye' : 5,
        'glasses' : 6,
        'left_ear' : 7,
        'right_ear' : 8,
        'nose' : 10,
        'mouth' : 11,
        'upper_lip': 12,
        'lower_lip': 13,
        'neck' : 14,
        'neck_l' : 15,
        'cloth' : 16,
        'hair': 17,
        'hat' : 18        
    }

    #intensity
    intensity = 0.1
    
    image_path = "ti.jpg"

    image = cv2.imread(image_path)
    ori = image.copy()
    
    #image = cv2.resize(image,(512,512))
    
    parsing = face_parser(image)

    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
    
    # [B, G, R]
    colors = [
        [230, 50, 20], [20, 70, 180], [100, 200, 100], [30, 250, 150], [250, 30, 150], [150, 250, 30],  [150, 50, 50], [50, 150, 50], [50, 50, 150],
        [80, 170, 255], [50, 200, 150], [60, 90, 230], [255, 255, 0], [255, 0, 255], [0, 255, 255], [230, 60, 90], [90, 230, 60], [130, 200, 255],
    ]
    parts = [
        table['face'], table['left_brow'], table['right_brow'], table['left_eye'], table['right_eye'], table['left_ear'], table['right_ear'], table['nose'],
        table['mouth'], table['upper_lip'], table['lower_lip'], table['neck'], table['neck_l'], table['neck_l'], table['cloth'], table['hair'], table['hat']
    ]

    for part, color in zip(parts, colors):
        image = make(image, parsing, part, color)

