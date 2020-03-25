from matplotlib import pyplot as plt
import imageio
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Maps and converts a segmentation image dataset to grayscale images"
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Image path, or a directory name"
    )
    args = parser.parse_args()
    image=imageio.imread(args.image)
    print(image)
    plt.axis('off')
    plt.imshow(image)
    plt.show()
