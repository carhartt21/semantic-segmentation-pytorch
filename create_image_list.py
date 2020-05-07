import os
import struct
import collections
import io
import argparse
import json
from utils import find_recursive


class UnknownImageFormat(Exception):
    pass


def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct builtin modules
    """
    size = os.path.getsize(file_path)
    with io.open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(26)
        if ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
                and (data[12:16] == b'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith(b'\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF):
                        b = input.read(1)
                    while (ord(b) == 0xFF):
                        b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(
                            int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
            except Exception as e:
                raise UnknownImageFormat(str(e))
    return width, height


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Creates a formattet list of input images and segmentations"
    )
    parser.add_argument(
        "--imgpath",
        required=True,
        type=str,
        help="base directory name"
    )
    parser.add_argument(
        "--segpath",
        default="data/",
        required=False,
        type=str,
        help="base directory name"
    )
    parser.add_argument(
        "--outfile",
        default="output/imageList.txt",
        type=str,
        help="path to output file",
        required=False
    )
    args = parser.parse_args()

    imgs = []
    if os.path.isdir(args.imgpath):
        imgs += find_recursive(args.imgpath, '.jpg')
        imgs += find_recursive(args.imgpath, '.png')
    else:
        print("Exception: imgpath {} is not a directory".format(args.imgpath))

    if not os.path.isdir(args.segpath):
        print("Exception: segpath {} is not a directory".format(args.segpath))
    print('{} images found in {}'.format(len(imgs), args.imgpath))
    #     print(args.segs)
    #     segs = find_recursive(args.segs)
    # else:
    list = []
    for img in imgs:
        imgSize = get_image_size(img)
        seg = img.replace('images', 'labels')
        seg = seg.replace('.jpg', '.png')
        if os.path.isfile(seg):
            list.append({'fpath_img': img, 'fpath_segm': seg, 'width': imgSize[0], 'height': imgSize[1]})
        else:
            print('Exception: could not find segmentation file {}'.format(seg))
    with open(args.outfile, 'w') as outfile:
        json.dump(list, outfile, indent=1, separators=(',', ':'))
    print('Finished: wrote {} files to file {}'.format(len(list), args.outfile))
