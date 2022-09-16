import os
from glob import glob
from collections import defaultdict
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from utils.latent_utils import attribute_label_from_segmentation
from configs.transforms_config import ToOneHot, Conver2Uint8, MyResize
from dataset.paired_dataset import PairedDataset


seg_transform = transforms.Compose([
    transforms.ToTensor(),
    Conver2Uint8(),
    MyResize((256, 256)),
    ToOneHot(5)
    ])

bscan_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 1, [0.5] * 1)
    ])

def group_images(image_dir):
    """
    Group all pictures in a folder by their name prefixes. Here we assume all images end with -xxx.png
    The returned dict values (list) are sorted
    """
    globs = glob(os.path.join(image_dir, '*'))
    groups = defaultdict(list)
    for image_path in globs:
        image_trunk = image_path[:-8].split('/')[-1]
        groups[image_trunk].append(image_path)
    for k in groups.keys():
        groups[k] = sorted(groups[k])
    return groups

def create_pairs(group, print_stats=False, sample_rate=0.1, positive_rate=0.8):
    """
    Create training samples within a group. It will n pairs, where
    n = generate sample_rate * n_instrument_images * n_images
    
    Args:
        group: a list of image paths
    Returns:
        attributes: [n_image, n_attribute]
        pairs: [n_pair, 2]
    """
    # 1. Calculate the attribute of each image
    attributes = []
    for img_path in group:
        seg = Image.open(img_path).convert('L')
        seg = seg_transform(seg)
        attributes.append(attribute_label_from_segmentation(seg, normalize=True))
    # 2. Sample image pairs according to the attributes
    attributes = np.array(attributes) # [n_samples, n_attributes]
    # for each attribute with instrument or 
    instrument_appeared = np.isclose(attributes[:, 0], 1)
    shadow_appeared = np.isclose(attributes[:, 5], 1)
    # For each picture with instrument, we sample 4/5 from others with instrument and sample 1/5 from
    # those without instrument
    instrument_or_shadow_appeared = np.logical_or(instrument_appeared, shadow_appeared)
    appear_list = np.where(instrument_or_shadow_appeared == True)[0]
    not_appear_list = np.where(instrument_or_shadow_appeared != True)[0]
    n_images = len(group)
    n_samples_per_image = int(sample_rate * n_images)
    n_positive = int(positive_rate * n_samples_per_image)
    n_negative = n_samples_per_image - n_positive
    pairs = []
    for idx in appear_list:
        positive_samples = np.random.choice(appear_list, n_positive, replace=False)
        negative_samples = np.random.choice(not_appear_list, n_negative, replace=False)
        to_pair = np.concatenate((positive_samples, negative_samples))
        idx_dup = np.full_like(to_pair, idx)
        paired = np.stack((idx_dup, to_pair)).T
        pairs.append(paired)
    # we also create a self transform
    to_pair = list(range(n_images))
    paired = np.stack((to_pair, to_pair)).T
    pairs.append(paired) 
    # 3. Return 2 lists: attributes, index pairs
    pairs = np.concatenate(pairs, axis=0) # pairs is of shape [n_pairs, 2]
    if print_stats:
        print(f"Instrument appeared: {np.sum(instrument_appeared)}, shadow appeared: \
{np.sum(shadow_appeared).item()}, total images: {attributes.shape[0]}, \
generated {pairs.shape[0]} pairs")
    return attributes, pairs


def prepare_paired_dataset():
    """
    A helper function that creates training dataset for paired latent transform task
    """
    seg_folder = '/home/extra/micheal/pixel2style2pixel/data/ioct/labels/train/'
    bscan_folder = '/home/extra/micheal/pixel2style2pixel/data/ioct/bscans/train/'
    groups = group_images(seg_folder)

    seg_paths = []
    all_attributes = []
    all_pairs = []
    for k in sorted(list(groups.keys())):
        group = groups[k]
        attributes, pairs = create_pairs(group, print_stats=True)
        seg_paths += group
        # Set an offset to the image index
        pairs += len(seg_paths)
        all_attributes.append(attributes)
        all_pairs.append(pairs)
    all_attributes = np.concatenate(all_attributes, axis=0)
    all_pairs = np.concatenate(all_pairs, axis=0)

    seg_paths_from_groups = sorted(list(glob(seg_folder + '*')))
    if seg_paths_from_groups != seg_paths:
        print(f"len seg_path {len(seg_paths)} len grouped seg path: {len(seg_paths_from_groups)}")
        raise ValueError("Mismatch")
    bscan_paths = sorted(glob(bscan_folder + '*'))
    train_dataset = PairedDataset(seg_paths=seg_paths, bscan_paths=bscan_paths, 
                                  attributes=all_attributes, seg_transform=seg_transform,
                                  bscan_transform=bscan_transform, pair_indices=all_pairs)
    print(f"len train dataset: {len(train_dataset)}")


if __name__ == '__main__':
    prepare_paired_dataset()
