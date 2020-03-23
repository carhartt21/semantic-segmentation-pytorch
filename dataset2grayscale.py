from matplotlib import pyplot as plt
import argparse
import os
import json
from utils import colorEncode, find_recursive

inputDir=''
outputDir=''
colorMapping={

}

classMapping={

}

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

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
        os.makedirs(cfg.TEST.result)
