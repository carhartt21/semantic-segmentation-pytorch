from dataset import TrainDataset
from config import cfg
from lib.nn import UserScatteredDataParallel, user_scattered_collate
import torch
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import json
from utils import colorEncode
import os
from lib.utils import as_numpy


def visualize_result(data, pred, cfg):
    colors = []
    names = {}
    inv_normalize = transforms.Normalize(
        mean=[-0.48 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    with open(cfg.DATASET.classInfo) as f:
        cls_info = json.load(f)
    for c in cls_info:
        names[c] = cls_info[c]['name']
        colors.append(cls_info[c]['color'])
    colors = np.array(colors, dtype='uint8')
    (img, info) = data
    # transform tensor to image data
    pred = as_numpy(pred.squeeze(0).cpu())
    pred = np.int32(pred)
    img = inv_normalize(img.squeeze())
    img = np.uint8(img * 255).transpose((1, 2, 0))
    uniques, counts = np.unique(pred, return_counts=True)
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    # aggregate images and save
    print(img.shape)
    img = img.resize(pred_color.shape[:1])
    im_vis = np.concatenate((img, pred_color), axis=1)
    img_name = 'batch_{}'.format(info)
    Image.fromarray(im_vis).save(os.path.join(cfg.TEST.result, '{}.png'.format(img_name)))

cfg.merge_from_file('config/outside30k-hrnetv2-ocr.yaml')
dataset_train = TrainDataset(
    cfg.DATASET.root_dataset,
    cfg.DATASET.list_train,
    cfg.DATASET,
    batch_per_gpu=cfg.TRAIN.batch_size_per_gpu,
    spatial=cfg.MODEL.spatial)

loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    shuffle=False,  # parameter is not used
    collate_fn=user_scattered_collate,
    num_workers=1,
    drop_last=True,
    pin_memory=True)
# create loader iterator
iterator_train = iter(loader_train)

for i in range(50):
    print(i)
    batch_data = next(iterator_train)
    batch_data = batch_data[0]
    visualize_result((batch_data['img_data'], i), batch_data['seg_label'], cfg)
