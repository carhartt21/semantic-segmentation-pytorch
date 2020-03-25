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


inputDir=''
outputDir=''
colorMappingFile=Path('data/colorsMapillary.json')
nameMappingFile=Path('data/mappingMapillary.json')

with open(nameMappingFile) as mfile:
    mapNames = json.load(mfile)
with open(colorMappingFile) as mfile:
    mapColors = json.load(mfile)

def remapImage(img, output):
    """Maps an image to grayscale image with new classes according to the maps.

    Parameters
    ----------
    img : np.array (m,n,o)
        Image data with semantic segmentation.
    mapColors : dict
        Maps color(RGB) to class of the input image.
    mapNames : dict
        Maps old class names to the new ones.
    output: string
        Output folder

    Returns
    -------
    type np.array (m,n,1)
        Grayscale image with semantic segmentation.

    """

    imgData = imageio.imread(img)
    grayImage = np.zeros(imgData.shape[:-1],dtype='uint8')
    #pbar = tqdm(total=imgData.shape[0]*imgData.shape[1], desc=img, ascii=True)
    for x in range(0,imgData.shape[0]):
        for y in range(0,imgData.shape[1]):
            newClass = -1
            try:
                oldClass = list(mapColors.values()).index(list(imgData[x][y][:-1]))
            except ValueError:
                print('Exception: class {} in {} at [{}, {}] not found'.format(imgData[x][y][-1], img, x, y))
            try:
                newClass = mapNames[str(oldClass)]
            except ValueError:
                print('Exception: no mapping for class {} at [{}, {}]'.format(oldClass, x, y))
            grayImage[x][y] = newClass
#                print('[{}, {}] : {}->{}'.format(x, y, oldClass, newClass))
            #pbar.update(1)
    imageio.imwrite('{}{}'.format(output, img.split('/')[-1]), grayImage)


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
        required=True,
        type=str,
        help="Path for output files"
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
    pool.starmap(remapImage(),[(img,args.output) for img in imgs])
    pool.close()
    # for img in imgs:
    #     remapImage(img, output=args.output)
