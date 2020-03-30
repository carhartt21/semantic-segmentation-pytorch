import numpy as np
from scipy import stats
from timeit import default_timer as timer
import imageio
import visvis as vv


def modNeighbors(i, j, d=1):
    n = segData[max(i-1,0):min(segData.shape[0]-1, i+2), max(0,j-1):min(segData.shape[1],j+2)].flatten()
    # remove the element (i,j)
    # n = np.hstack((b[:len(b)//2],b[len(b)//2+1:] ))
    return stats.mode(n,axis=None)[0]

segData = imageio.imread('output/ADE_train_00020210_new.png')
# segData = np.array([
#       [ 11,  21,  31,  41,  51,  61,  71],
#       [ 21,  np.nan,  32,  42,  52,  62,  72],
#       [ 13,  23,  33,  np.nan,  53,  63,  73],
#       [ 14,  24,  34,  44,  54,  64,  74],
#       [ 15,  25,  35,  45,  74,  np.nan,  75],
#       [ 16,  26,  36,  46,  56,  66,  76],
#       [ 17,  27,  37,  47,  57,  67,  77]])

print(segData)
iX, iY = np.nonzero((segData==0))
for x,y in zip(iX, iY):
    segData[x][y] = modNeighbors(x,y)

imageio.imwrite('test.png',segData)
print(segData)
