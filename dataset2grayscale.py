from matplotlib import pyplot as plt
import argparse
import os
import json
from utils import colorEncode, find_recursive

inputDir=''
outputDir=''
colorMapMappilary=[
        [165, 42, 42]
        [0, 192, 0]
        [196, 196, 196]
        [190, 153, 153]
        [180, 165, 180]
        [90, 120, 150]
        [102, 102, 156]
        [128, 64, 255]
        [140, 140, 200]
        [170, 170, 170]
        [250, 170, 160]
        [96, 96, 96]
        [230, 150, 140]
        [128, 64, 128]
        [110, 110, 110]
        [244, 35, 232]
        [150, 100, 100]
        [70, 70, 70]
        [150, 120, 90]
        [220, 20, 60]
        [255, 0, 0]
        [255, 0, 100]
        [255, 0, 200]
        [200, 128, 128]
        [255, 255, 255]
        [64, 170, 64]
        [230, 160, 50]
        [70, 130, 180]
        [190, 255, 255]
        [152, 251, 152]
        [107, 142, 35]
        [0, 170, 30]
        [255, 255, 128]
        [250, 0, 30]
        [100, 140, 180]
        [220, 220, 220]
        [220, 128, 128]
        [222, 40, 40]
        [100, 170, 30]
        [40, 40, 40]
        [33, 33, 33]
        [100, 128, 160]
        [142, 0, 0]
        [70, 100, 150]
        [210, 170, 100]
        [153, 153, 153]
        [128, 128, 128]
        [0, 0, 80]
        [250, 170, 30]
        [192, 192, 192]
        [220, 220, 0]
        [140, 140, 20]
        [119, 11, 32]
        [150, 0, 255]
        [0, 60, 100]
        [0, 0, 142]
        [0, 0, 90]
        [0, 0, 230]
        [0, 80, 100]
        [128, 64, 64]
        [0, 0, 110]
        [0, 0, 70]
        [0, 0, 192]
        [32, 32, 32]
        [120, 10, 10]
        [0, 0, 0]
        ]

classMapMappillary={

}

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required

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
        imgData = plt.imread(img)
        for x,y in imgData:
            # mapp color to class
            if img[x][y] = colorMap

    plt.imshow(grayImage)
    plt.show()
    json.load()
