from utils import colorEncode, find_recursive
from config import cfg
from PIL import Image
import numpy as np
import os
import argparse
import json

def visualize_result(image_path, label_path, dir_result, colors):
    # segmentation
    img = np.asarray(Image.open(image_path).convert('RGB'))
    seg = np.asarray(Image.open(label_path)).copy()
    seg -= 1
    seg_color = colorEncode(seg, colors)
    # aggregate images and save
    im_vis = np.concatenate((img, seg_color),
                            axis=0).astype(np.uint8)
    img_name = image_path.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))
    print('Saved image at: {}'.format(os.path.join(dir_result, img_name.replace('.jpg', '.png'))))



if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(description='Calculate per-class statistics from configuration file')
    parser_.add_argument('--cfg', nargs='?', type=str, default='',
                         help='Configuration file to calculate statistics for')
    parser_.add_argument('--input', nargs='?', type=str,
                         help='Name of input folder')
    parser_.add_argument('--output', nargs='?', type=str, default='color_encoded_labels',
                         help='Name of output folder')
    args = parser_.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    imgs = []
    if os.path.isdir(args.input):
        imgs += find_recursive(args.input, '.jpg')
    else:
        print("Exception: imgpath {} is not a directory".format(args.input))
    colors = []
    with open('data/outside30k.json') as f:
        cls_info = json.load(f)
    for c in cls_info:
        colors.append(cls_info[c]['color'])
    for img in imgs:
        visualize_result(img, img.replace('.jpg', '.png'), args.output, colors)
