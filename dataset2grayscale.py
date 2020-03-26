import imageio
import numpy as np
import argparse
import os
import json
import multiprocessing as mp
import time

from pathlib import Path
from utils import colorEncode, find_recursive
from tqdm import tqdm


colorMappingFile=Path('data/colorsMapillary.json')
nameMappingFile=Path('data/mappingMapillary.json')

with open(nameMappingFile) as mfile:
    mapNames = json.load(mfile)
with open(colorMappingFile) as mfile:
    mapColors = list(json.load(mfile).values())

def remapImage(img):
    """Maps an image to grayscale image with new classes according to the maps.

    Parameters
    ----------
    img : np.array (m,n,o)
        Image data with semantic segmentation.

    """
#    output=Path('/media/chge7185/HDD1/datasets/mapillary/new_labels/')
    output=Path('/mnt/Data/chge7185/datasets/new_labels/')
    imgData = imageio.imread(img)
    grayImage = np.zeros(imgData.shape[:-1],dtype='uint8')
    imgName = img.split('/')[-1]
    if os.path.isfile('{}/{}'.format(output, imgName)):
        return
    for x in range(0,imgData.shape[0]):
        for y in range(0,imgData.shape[1]):
            try:
                oldClass = mapColors.index(list(imgData[x][y][:-1]))
            except ValueError:
                print('Exception: class {} in {} at [{}, {}] not found'.format(imgData[x][y][-1], img, x, y))
            try:
                grayImage[x][y] = mapNames[str(oldClass)]
            except ValueError:
                print('Exception: no mapping for class {} at [{}, {}]'.format(oldClass, x, y))
    imageio.imwrite('{}/{}'.format(output, img.split('/')[-1]), grayImage)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Maps and converts a segmentation image dataset to grayscale images"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Image path, or a directory name"
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        help="Path for output files",
        default='output/'
    )
    parser.add_argument(
        "--dataset",
        required=False,
        type=str,
        help="Dataset type",
        default='mapillary'
    )
    args = parser.parse_args()
    # generate image list
    if os.path.isdir(args.input):
        print(args.input)
        imgs = find_recursive(args.input, ext='.png')
    else:
        imgs = [args.input]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm(pool.imap_unordered(remapImage,[(img) for img in imgs], chunksize=10), total=len(imgs), desc='Mapping images', ascii=True):
       pass
    pool.close()
