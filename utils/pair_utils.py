import os
from os.path import exists
from pathlib import Path
from glob import glob
from collections import defaultdict
import pickle
from argparse import Namespace
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torcheck
from utils.latent_utils import attribute_label_from_segmentation, get_latent, modify_attribute
from configs.transforms_config import ToOneHot, Convert2Uint8, MyResize
from configs.data_configs import DATASETS
from dataset.paired_dataset import PairedDataset



# seg_transform = transforms.Compose([
#     transforms.ToTensor(),
#     Convert2Uint8(),
#     MyResize((256, 256)),
#     ToOneHot(5)
#     ])

# bscan_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5] * 1, [0.5] * 1)
#     ])

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


def create_pairs(group, seg_transform, print_stats=False):
    """
    Create training samples within a group. It will n pairs, where
    n = n_instrument_or_shadow_appeared ^ 2 + n_images

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
    attributes = np.asarray(attributes) # [n_samples, n_attributes]
    # for each attribute with instrument or 
    instrument_appeared = np.isclose(attributes[:, 0], 1)
    shadow_appeared = np.isclose(attributes[:, 5], 1)
    # For each picture with instrument, we sample 4/5 from others with instrument and
    # sample 1/5 from those without instrument
    instrument_or_shadow_appeared = np.logical_or(instrument_appeared, shadow_appeared)
    appear_list = np.where(instrument_or_shadow_appeared == True)[0]
    not_appear_list = np.where(instrument_or_shadow_appeared != True)[0]
    pairs = []
    positive_samples = [] # instruments appear in both images
    # for each image with instruments, pair it with every other image with instruments
    for idx_src in appear_list:
        for idx_dst in appear_list:
            positive_samples.append((idx_src, idx_dst))
    
    negative_samples = [] # self-pairing
    for idx_src in np.concatenate((not_appear_list, appear_list)):
        negative_samples.append((idx_src, idx_src))
    
    pairs = positive_samples + negative_samples
    # 3. Return 2 lists: attributes, index pairs
    pairs = np.asarray(pairs) # pairs is of shape [n_pairs, 2]
    if print_stats:
        print(f"Instrument appeared: {np.sum(instrument_appeared)}, shadow appeared: \
{np.sum(shadow_appeared).item()}, total images: {attributes.shape[0]}, \
generated {pairs.shape[0]} pairs")
    return attributes, pairs


def prepare_paired_dataset(seg_folder = 'data/ioct/labels/train/',
                           bscan_folder = 'data/ioct/bscans/train/',
                           attributes_cache_dir='artifacts/cache/paired',
                           dataset_name='ioct_amd_seg_to_bscan',
                           label_nc=6,
                           output_nc=1,
                           print_stats=False):
    """
    A helper function that creates training dataset for paired latent transform task.
    Each sample of the dataset contains (src_seg, src_bscan, src_attr, dst_seg, dst_bscan,
    dst_attr)

    Args:
        attributes_cache_dir: the directory to cache attributes and pairs
        dataset_name: will be used to locate transform functions
        label_nc: number of label channels of the segmentation
        output_nc: number of output channels of the bscan
    Returns:
        train_dataset: training dataset
    """
    transform_config = Namespace(label_nc=label_nc, output_nc=output_nc)
    transform_dict = DATASETS[dataset_name]['transforms'](transform_config).get_transforms()
    seg_transform = transform_dict['transform_source']
    bscan_transform = transform_dict['transform_gt_train']

    groups = group_images(seg_folder)
    if print_stats:
        print(f"Found {len(groups)} groups, their respective image prefixes are: {sorted(list(groups.keys()))}")
    # Load all seg paths for comparison with the saved ones
    seg_paths = []
    for k in sorted(list(groups.keys())):
        seg_paths += groups[k]

    load_saved = False

    # Define save paths
    if attributes_cache_dir is not None:
        Path(attributes_cache_dir).mkdir(parents=True, exist_ok=True)
        all_pairs_path = os.path.join(attributes_cache_dir, 'all_pairs.npy')
        all_attributes_path = os.path.join(attributes_cache_dir, 'all_attributes.npy')
        seg_paths_path = os.path.join(attributes_cache_dir, 'seg_paths.pkl')
        # Check whether the saved ones are qualified
        if exists(all_pairs_path) and exists(all_attributes_path) and exists(seg_paths_path):
            saved_seg_paths = pickle.load(open(seg_paths_path, 'rb'))
            if saved_seg_paths == seg_paths:
                load_saved = True
                print("Loading attributes from cache...")
                with open(all_pairs_path, 'rb') as f:
                    all_pairs = np.load(f)
                with open(all_attributes_path, 'rb') as f:
                    all_attributes = np.load(f)

    # Attain all attributes if there isn't a saved copy
    if not load_saved:
        all_attributes = []
        all_pairs = []
        seg_paths = [] # we need this again to count accumulated segmentations
        for k in sorted(list(groups.keys())):
            group = groups[k]
            attributes, pairs = create_pairs(group, seg_transform, print_stats=print_stats)
            # Set an offset to the image index
            pairs += len(seg_paths)
            # Update seg_paths AFTER setting the offset
            seg_paths += group
            all_attributes.append(attributes)
            all_pairs.append(pairs)
        all_attributes = np.concatenate(all_attributes, axis=0)
        all_pairs = np.concatenate(all_pairs, axis=0)
        # Save computation results
        if attributes_cache_dir is not None:
            with open(seg_paths_path, 'wb') as f:
                pickle.dump(seg_paths, f)
            with open(all_attributes_path, 'wb') as f:
                np.save(f, all_attributes)
            with open(all_pairs_path, 'wb') as f:
                np.save(f, all_pairs)
            print(f"Saved attributes etc. cache to {attributes_cache_dir}")

    # Patch: attributes should be of float 32, or there will be type mismatch
    # in model forward
    all_attributes = all_attributes.astype(np.float32)
    # This is to conform to the assumption that:
    # 1. The path groups covers all samples in the segmentation folder
    # 2. The sorted prefix+suffix conforms with the sorted files in the folder
    seg_paths_from_groups = sorted(list(glob(seg_folder + '*')))
    if seg_paths_from_groups != seg_paths:
        print(f"len seg_path {len(seg_paths)} len grouped seg path: {len(seg_paths_from_groups)}")
        raise ValueError("Mismatch")
    # Create a dataset from the attributes and pairs
    bscan_paths = sorted(glob(bscan_folder + '*'))
    paired_dataset = PairedDataset(seg_paths=seg_paths, bscan_paths=bscan_paths,
                                  attributes=all_attributes, seg_transform=seg_transform,
                                  bscan_transform=bscan_transform, pair_indices=all_pairs)
    print(f"len dataset: {len(paired_dataset)}")
    return paired_dataset


def train_pairs_batch(src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr,
                      style_model, latent_model, device):
    """
    Train the latent to latent models with paired data.

    Args:
        src_seg: source segmentation
        src_bscan: source bscan
        src_attr: source attributes
        dst_seg: destination segmentation
        dst_bscan: destination bscan
        dst_attr: destination attributes
        style_model: style model (frozen)
        latent_model: the latent model to train
        device: device to run on
    Returns:
        loss: loss value
    """
    with torch.no_grad():
        _, _, w_latents_src = get_latent(style_model, src_seg, device)
        _, _, w_latents_dst = get_latent(style_model, dst_seg, device)
    
    # the appearance of the instruments and shadows (are the same as dst)
    delta_attributes = dst_attr - src_attr
    # delta_attributes[:, 0] = dst_attr[:, 0]
    # delta_attributes[:, 5] = dst_attr[:, 5]
    delta_attributes_backward = -delta_attributes
    # delta_attributes_backward[:, 0] = src_attr[:, 0]
    # delta_attributes_backward[:, 5] = src_attr[:, 5]

    w_n = latent_model(w_latents_src, delta_attributes)

    # Target losss: converted latent resemble target latent
    loss_dst = F.mse_loss(w_n, w_latents_dst)

    # Cycle loss: when convert the latent backword with the negative
    # delta attributes, the latent should be the same as the original
    w_n_c = latent_model(w_n, delta_attributes_backward)
    loss_cycle = F.mse_loss(w_n_c, w_latents_src)

    # Identity loss: can be omitted, while the self-transform is included in
    # the training set
    w_i = latent_model(w_latents_src, 0 * delta_attributes)
    loss_identity = F.mse_loss(w_i, w_latents_src)

    # Neighborhood loss is omitted, similar constraint is enforced with target losss
    # loss_neighborhood = F.mse_loss(w_n_forward, w_latents_src)

    # Reconstruction loss: use the generted dst latent to reconstruct the dst bscan,
    # the bscan should resemble the target bscan
    generated_images_dst = style_model(w_n, input_code=True)

    # Reduction is default to meanx, which means it doesn't differ with previous
    # losses in scale
    loss_reconstruct = F.mse_loss(generated_images_dst, dst_bscan)

    loss = loss_dst * 5 + loss_cycle + loss_identity + loss_reconstruct
    return loss


def load_coach(model_path='/home/extra/micheal/pixel2style2pixel/experiments/ioct_seg2bscan2/checkpoints/best_model.pt',
              batch_size=16, load_partial_weights=True):
    """
    Load Coach object from pretraining. This is used to simplify the
    boilderplate code for loading the style model.

    Args:
        model_path: path to the trained checkpoint
        batch_size: batch size for the dataloader (used for style model)
        load_partial_weights: if True, only load the weights for the style model
    Returns:
        Coach object
    """
    from training.coach import Coach
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    optss = Namespace(**opts)
    optss.batch_size = batch_size
    optss.stylegan_weights = model_path
    optss.load_partial_weights = load_partial_weights
    return Coach(optss)


class InferenceGenerator:
    """
    A inference time helper class that generates images based on modified attributes.
    """
    def __init__(self, style_model, latent_model, correlation_matrix, device):
        self.style_model = style_model
        self.latent_model = latent_model
        self.device = device
        self.correlation_matrix = correlation_matrix
        
    def generate_new(self, w_latents, attribute, change, change_channel):
        modified_attribute, actual_change = modify_attribute(attribute, self.correlation_matrix, 
                                                             change_channel=change_channel,
                                                             change=change)
        modified_attribute = torch.Tensor(modified_attribute).unsqueeze(0)
        attribute = torch.Tensor(attribute).unsqueeze(0)
        delta_attributes = modified_attribute - attribute
        delta_attributes = delta_attributes.to(self.device)
        w_latents = w_latents.to(self.device)
        with torch.no_grad():
            w_n =  self.latent_model(w_latents, delta_attributes)
            w_n = 0.7*w_latents + 0.3*w_n
            generated_images = self.style_model(w_n, input_code=True).detach().cpu().numpy()
        generated_image = generated_images[0][0]
        return generated_image, actual_change

    def generate_new_imgs(self, seg, changes, change_channel=1):
        segs = seg.unsqueeze(0).float().cuda()
        with torch.no_grad():
            _, w_latents, w_codes = get_latent(self.style_model, segs, self.device)
        attribute = attribute_label_from_segmentation(seg, normalize=True)
        results = []
        for change in changes:
            img, actual_change = self.generate_new(w_latents, attribute, change, change_channel)
            results.append((img, actual_change))
        return results

    def generate_original(self, seg):
        segs = seg.unsqueeze(0).float().cuda()
        with torch.no_grad():
            _, w_latents, w_codes = get_latent(self.style_model, segs, self.device)
        w_latents = w_latents.to(self.device)
        reconstructed = self.style_model(w_latents, input_code=True).detach().cpu().numpy()
        return reconstructed[0][0]



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from models.latent2latent import Latent2Latent
    # dataset = prepare_paired_dataset()
    dataset = prepare_paired_dataset(seg_folder = 'data/overfit/labels/train/',
        bscan_folder = 'data/overfit/bscans/train/',
        attributes_cache_dir='artifacts/cache/paired_overfit',
        print_stats=True)
    dataloader = DataLoader(dataset, batch_size=16)
    batch = next(iter(dataloader))
    src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr = batch
    # load style model
    STYLE_MODEL_PATH = 'experiments/ioct_seg2bscan5/checkpoints/best_model.pt'
    coach = load_coach(STYLE_MODEL_PATH, batch_size=4)
    style_model = coach.net
    # Initialize latent2latent model
    device = torch.device('cuda')
    latent_model = Latent2Latent().to(device)
    style_model = style_model.to(device)
    for p in style_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(latent_model.parameters(), lr=1e-2)
    torcheck.register(optimizer)
    torcheck.add_module_changing_check(latent_model, module_name='latent2latent')
    # torcheck.add_module_unchanging_check(style_model, module_name='stylegan')
    torcheck.verbose_on()
    style_model.latent_avg = style_model.latent_avg.to(device)
    src_seg, src_bscan, src_attr = src_seg.to(device), src_bscan.to(device), src_attr.to(device)
    dst_seg, dst_bscan, dst_attr = dst_seg.to(device), dst_bscan.to(device), dst_attr.to(device)

    for _ in range(3):
        loss = train_pairs_batch(src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr,
                                    style_model, latent_model, device)
        print('loss:', loss)
        loss.backward()
        optimizer.step()
