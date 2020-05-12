import argparse
import logging
import sys
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import torch
import torch.utils
from math import ceil
from dataset import BaseDataset, StatsDataset
from config import cfg

log = logging.getLogger(__name__)
# global plt settings
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'x'
plt.rcParams['grid.linestyle'] = 'dashed'

def stats(args):
    cfg.merge_from_file(args.cfg)
    data_loader = StatsDataset(
        cfg.DATASET.root_dataset, cfg.DATASET.list_stats, cfg.DATASET)

    class_colors = []
    class_names = []
    assert os.path.isfile(cfg.DATASET.classInfo), 'Class information file is missing'
    with open(cfg.DATASET.classInfo) as f:
        cls_info = json.load(f)
    for c in cls_info:
        class_names.append(cls_info[c]['name'])
        class_colors.append(cls_info[c]['color'])
    log.info('Loading data from: {}'.format(cfg.DATASET.list_stats))
    torch_data_loader = torch.utils.data.DataLoader(data_loader, batch_size=1, num_workers=16, shuffle=False)

    num_classes = cfg.DATASET.num_class
    num_images = len(data_loader)
    class_image_count = np.zeros(num_classes)
    class_image_percentage = np.zeros(num_classes)
    class_pixel_count = np.zeros(num_classes)
    class_pixel_percentage = np.zeros(num_classes)
    class_percentage_per_image = np.zeros(num_classes)
    class_heatmap = np.zeros((num_classes, 100, 100))

    log.info('Number of images: {0}'.format(num_images))
    log.info('Number of classes: {0}'.format(num_classes))

    for i, (lbl, lbl_) in tqdm.tqdm(enumerate(torch_data_loader), total=len(data_loader), ascii=True):
        labels = np.squeeze(lbl.numpy())
        labels_ = np.squeeze(lbl_.numpy())
        pixel_count_ = np.bincount(labels.flatten(), minlength=num_classes)
        for c in range(num_classes):
            if pixel_count_[c] > 0:
                class_image_count[c] += 1.0
                class_percentage_per_image[c] += pixel_count_[c] / labels.size
                class_pixel_count[c] += pixel_count_[c]
                class_heatmap[c][labels_ == c] += 1.0

        log.debug(class_pixel_count)
        log.debug(class_image_count)

    for c in range(num_classes):
        class_image_percentage[c] = class_image_count[c] * 100.0 / num_images
        class_pixel_percentage[c] = class_pixel_count[c] * 100.0 / np.sum(class_pixel_count)
        if class_image_count[c]:
            class_percentage_per_image[c] = class_percentage_per_image[c] * 100.0 / class_image_count[c]
        # Normalize heatmap
        if class_heatmap[c].max():
            class_heatmap[c] /= np.max(class_heatmap[c])

    log.debug('Sum of percentages of images: {0}'.format(np.sum(class_image_percentage)))
    log.debug('Sum of percentages of pixels: {0}'.format(np.sum(class_pixel_percentage)))
    log.info("Summary:")
    for c in range(num_classes):
        log.info('Class {} appears in {:d} images ({:.2f}%), totalling {:d} pixels ({:.2f}%)'.format(
            c,
            int(class_image_count[c]),
            class_image_percentage[c],
            int(class_pixel_count[c]),
            class_pixel_percentage[c]))

    # Plot per-class image percentage
    x_labels_ = range(0, num_classes)
    x_values_ = np.arange(len(x_labels_))
    y_values_ = class_image_percentage
    y_label_ = 'Percentage of Images (%)'
    fig, ax = plt.subplots()
    barlist = ax.barh(x_values_, y_values_)
    for c in range(num_classes):
        barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    ax.set_xlim(0, 100)
    ax.set_xlabel(y_label_)
    ax.set_yticks(x_values_)
    ax.set_yticklabels(x_labels_)
    ax.set_xlim(0, 5 * ceil(class_image_percentage.max() / 5.0))
    ax.invert_yaxis()
    ax.set_title('Per-class Image Percentage')
    plt.gca().margins(y=0.01)
    plt.gcf().set_size_inches(
        plt.gcf().get_size_inches()[0], 0.17 * num_classes)
    fig.savefig(os.path.join(args.out, 'class_dist.png'))

    # Plot per-class pixel percentage
    x_labels_ = range(0, num_classes)
    x_values_ = np.arange(len(x_labels_))
    y_values_ = class_pixel_percentage
    y_label_ = 'Percentage of Pixels (%)'
    fig, ax = plt.subplots()
    barlist = ax.barh(x_values_, y_values_)
    for c in range(num_classes):
        barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    ax.set_yticks(x_values_)
    ax.set_yticklabels(x_labels_)
    ax.invert_yaxis()
    # ax.set_xlim(0, class_pixel_percentage.max())
    ax.set_xlabel(y_label_)
    ax.set_xlim(0, 5 * ceil(class_pixel_percentage.max() / 5.0))
    ax.set_title('Per-class Pixel Percentage')
    plt.gca().margins(y=0.01)
    plt.gcf().set_size_inches(
        plt.gcf().get_size_inches()[0], 0.17 * num_classes)
    fig.savefig(os.path.join(args.out, 'pixel_dist.png'))

    # Plot per-class pixel percentage
    x_labels_ = range(0, num_classes)
    x_values_ = np.arange(len(x_labels_))
    y_values_ = class_percentage_per_image
    y_label_ = 'Percentage of Pixels (%)'
    fig, ax = plt.subplots()
    barlist = ax.barh(x_values_, y_values_)
    for c in range(num_classes):
        barlist[c].set_color(tuple(np.array(class_colors[c]) / 255.0))
    ax.set_yticks(x_values_)
    ax.set_yticklabels(x_labels_)
    ax.invert_yaxis()
    # ax.set_xlim(0, class_percentage.max())
    ax.set_xlabel(y_label_)
    ax.set_title('Average Image Per-class Pixel Percentage')
    ax.set_ylim(0, 5 * ceil(class_percentage_per_image.max() / 5.0))
    plt.gca().margins(y=0.01)
    plt.gcf().set_size_inches(
        plt.gcf().get_size_inches()[0], 0.17 * num_classes)
    fig.savefig(os.path.join(args.out, 'image_pixel_dist.png'))

    # Plot heatmaps
    for c in range(num_classes):
        fig, ax = plt.subplots()
        # red-blue palette
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(class_heatmap[c], ax=ax, vmin=0.0, vmax=1.0)
        ax.set_title('{0} heatmap'.format(class_names[c]))
        plt.axis('off')
        fig.savefig(os.path.join(args.out, 'heatmap_{0}.png'.format(c)))
        plt.close()

    np.savez(
        os.path.join(args.out, 'results'),
        class_pixel_percentage=class_pixel_percentage,
        class_image_percentage=class_image_percentage,
        class_percentage_per_image=class_percentage_per_image,
        class_heatmap=class_heatmap)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser_ = argparse.ArgumentParser(description='Calculate per-class statistics from configuration file')
    parser_.add_argument('--cfg', nargs='?', type=str, default='',
                         help='Configuration file to calculate statistics for')
    parser_.add_argument('--out', nargs='?', type=str, default='plots',
                         help='Name of output folder')
    args = parser_.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)                     
    stats(args)
