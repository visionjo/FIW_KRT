#!/usr/bin/env python
import os.path
import sys
from time import strftime
from PIL import Image
import argparse
import glob

row_size = 4
margin = 3


def generate_montage(filenames, output_fn):
    images = [Image.open(filename) for filename in filenames]

    width = max(image.size[0] + margin for image in images) * row_size
    height = sum(image.size[1] + margin for image in images)
    montage = Image.new(mode='RGBA', size=(width, height), color=(0, 0, 0, 0))

    max_x = 0
    max_y = 0
    offset_x = 0
    offset_y = 0
    for i, image in enumerate(images):
        montage.paste(image, (offset_x, offset_y))

        max_x = max(max_x, offset_x + image.size[0])
        max_y = max(max_y, offset_y + image.size[1])

        if i % row_size == row_size - 1:
            offset_y = max_y + margin
            offset_x = 0
        else:
            offset_x += margin + image.size[0]

    montage = montage.crop((0, 0, max_x, max_y))
    montage.save(output_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate image montage of N images from each directory',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, help='Input file or list of files.')
    # parser.add_argument('output', type=str, help='File path of output montage')
    parser.add_argument('-n', '--Number', type=int, default=3,
                        help='Number of images to include from each directory')
    args = parser.parse_args()

    basename = strftime("Montage %Y-%m-%d at %H.%M.%S.png")
    exedir = os.path.dirname(os.path.abspath(sys.argv[0]))
    filename = os.path.join(exedir, basename)
    filepaths = glob.glob(args.input)
    import pdb

    pdb.set_trace()
    generate_montage(filepaths, filename)
