import imageio
import numpy as np
import argparse
import os
import json
import progressbar
import time
from pathlib import Path
from utils import colorEncode, find_recursive

inputDir=''
outputDir=''
colorMappingFile=Path('data/colorsMapillary.json')
nameMappingFile=Path('data/mappingMapillary.json')

with open(nameMappingFile) as mfile:
    mapNames = json.load(mfile)
with open(colorMappingFile) as mfile:
    mapColors = json.load(mfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Maps and converts a segmentation image dataset to grayscale images"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="an path for output files"
    )
    args = parser.parse_args()

    # generate image list
    if os.path.isdir(args.input):
        print(args.imgs)
        imgs = find_recursive(args.input)
    else:
        imgs = [args.input]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    for img in imgs:
        print('{}')
        imgData = imageio.imread(img)
        grayImage = np.zeros(imgData.shape)
        for x in range(0,imgData.shape[0]):
            for y in range(0,imgData.shape[1]):
                newClass = -1
                try:
                    oldClass = list(mapColors.values()).index(list(imgData[0][0][:-1]))
                except ValueError:
                    print('Exception: class {} in {} at [{}, {}] not found'.format(imgData[x][y][-1], img, x, y))
                try:
                    newClass = mapNames[str(oldClass)]
                except ValueError:
                    print('Exception: no mapping for class {} at [{}, {}]'.format(oldClass, x, y))
                grayImage[x][y] = newClass
        imageio.imwrite('{}/{}'.format(args.output, img))
