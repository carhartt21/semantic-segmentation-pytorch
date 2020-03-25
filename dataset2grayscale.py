from matplotlib import pyplot as plt
import argparse
import os
import json
from pathlib import Path
from utils import colorEncode, find_recursive

inputDir=''
outputDir=''
colorMappingFile=Path('data/colorsMapillary.json')
nameMappingFile=Path('data/mappingMappilary.json')

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
    # generate image list
    if os.path.isdir(args.input):
        print(args.imgs)
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for img in imgs:
        imgData = plt.imread(img,'uint8')
        grayImage = np.zeros(imgData.shape)
        for x in range(0,imgData.shape[0])
            for y in range(0,imgData.shape[1]):
                try:
                    oldClass = list(mapColors.values()).index(list(imgData[0][0][:-1]))
                except ValueError:
                    print('Exception: class {} in {} at [{}, {}] not found'.format(imgData[x][y][-1], img, x, y))
                try:
                    newClass = list(mapNames.values()).index(oldClass)
                except ValueError:
                    print('Exception: no mapping for class'.format(oldClass))
                grayImage[x][y] = newClass
    plt.imshow(grayImage)
    plt.show()
    json.load()
