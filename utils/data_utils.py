"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(data_dir):
    """
    Parameters:
    ---------------------------------------------------------------------------
    dir: path to dataset folder. If multiple directories are provided, they are
        seperated by a comma; the files in these directories are merged.
    """
    images = []
    if ',' in data_dir:
        dirs = data_dir.split(',')
    else:
        dirs = [data_dir]
    for folder in dirs:
        assert os.path.isdir(folder), '%s is not a valid directory' % folder
        for root, _, fnames in sorted(os.walk(folder)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images
