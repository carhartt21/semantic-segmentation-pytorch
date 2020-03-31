import numpy as np
from scipy import stats
from timeit import default_timer as timer
import imageio
import argparse
import multiprocessing as mp
import os
from utils import find_recursive
from tqdm import tqdm

def modNeighbors(segData, i, j, d=1):
    n = segData[max(i-d,0):min(segData.shape[0]-d, i+d+1), max(0,j-d):min(segData.shape[1],j+d+1)].flatten()
    return stats.mode(n,axis=None)[0]

def smoothSegmentation(img):
    segData = imageio.imread(img)
    iX, iY = np.nonzero(segData==0)
    for x,y in zip(iX, iY):
        mod = modNeighbors(segData, x, y, d=2)
        # while mod == np.nan:
        #      d+=1
        #      mod = modNeighbors(segData, x,y,d)
        segData[x][y] = mod
    imageio.imwrite('{}/{}'.format(args.output, img.split('/')[-1]), segData)

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
    # Generate image list
    if os.path.isdir(args.input):
        print(args.input)
        imgs = find_recursive(args.input, ext='.png')
    else:
        imgs = [args.input]
    assert len(imgs), "Exception: imgs should be a path to image (.png|jpg) or directory."
    # Create output directory
    if not os.path.isdir(args.output):
        print('Creating empty output directory {}'.format(args.output))
        os.makedirs(args.output)
    pool = mp.Pool(args.nproc)
    # Assign tasks to workers
    for _ in tqdm(pool.imap_unordered(smoothSegmentation,[(img) for img in imgs], chunksize=args.chunk), total=len(imgs), desc='Smoothing segmentation images', ascii=True):
       pass
    # Close pool
    pool.close()
