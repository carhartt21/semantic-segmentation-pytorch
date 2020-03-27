import imageio
import numpy as np
import argparse
import os
import json
import multiprocessing as mp
import time
from tqdm import tqdm
from pathlib import Path
#internal libraries
from utils import colorEncode, find_recursive


def remapImage(img):
    """Maps an image to a grayscale image according to the maps and saves the result.

    Parameters
    ----------
    img : np.array (m,n,o)
        Image data with semantic segmentation.

    """
    # Read image
    imgData = imageio.imread(img)
    grayImage = np.zeros(imgData.shape[:-1],dtype='uint8')
    imgName = img.split('/')[-1]
    # Check if file exists already in the output path
    if os.path.isfile('{}/{}'.format(args.output, imgName)):
        return
    # Loop through pixels
    for x in range(0,imgData.shape[0]):
        for y in range(0,imgData.shape[1]):
            # Determine old class
            if args.dataset == 'mapillary':
                try:
                    oldClass = mapColors.index(list(imgData[x][y][:-1]))
                except ValueError:
                    print('Exception: class {} in {} at [{}, {}] not found'.format(imgData[x][y][-1], img, x, y))
            elif args.dataset == 'ADE20K':
                oldClass = imgData[x][y]
            # Map to new class
            try:
                grayImage[x][y] = mapNames[str(oldClass)]
            except ValueError:
                print('Exception: no mapping for class {} at [{}, {}]'.format(oldClass, x, y))
    # Save image
    imageio.imwrite('{}/{}'.format(args.output, img.split('/')[-1]), grayImage)
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
    parser.add_argument(
        "--nproc",
        required=False,
        type=int,
        help="Number of parralel processes",
        default=mp.cpu_count()
    )
    parser.add_argument(
        "--chunk",
        required=False,
        type=int,
        help="Chunk size for each worker thread",
        default=mp.cpu_count()
    )
    # Read args
    args = parser.parse_args()
    if args.dataset == 'mapillary':
        colorMappingFile = Path('data/colorsMapillary.json')
        nameMappingFile = Path('data/mappingMapillary.json')
        with open(colorMappingFile) as mfile:
            mapColors = list(json.load(mfile).values())
        with open(nameMappingFile) as mfile:
            mapNames = json.load(mfile)
    elif args.dataset == 'ADE20K':
        nameMappingFile = Path('data/mappingADE.json')
        with open(nameMappingFile) as mfile:
            mapNames = json.loads(mfile)
    else:
        print('Exception: Dataset type {} unknown'.format(dataset))
    # Generate image list
    if os.path.isdir(args.input):
        print(args.input)
        imgs = find_recursive(args.input, ext='.png')
    else:
        imgs = [args.input]
    assert len(imgs), "Exception: imgs should be a path to image (.jpg) or directory."
    # Create output directory
    if not os.path.isdir(args.output):
        print('Creating empty output directory {}'.format(output))
        os.makedirs(args.output)
    # Create worker pool
    pool = mp.Pool(args.nproc)
    # Assign tasks to workers
    for _ in tqdm(pool.imap_unordered(remapImage,[(img) for img in imgs], chunksize=args.chunk), total=len(imgs), desc='Mapping images', ascii=True):
       pass
    # Close pool
    pool.close()
