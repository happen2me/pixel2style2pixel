import numpy as np
import torch
from argparse import Namespace
from training.coach import Coach


def get_coor_avg(label_map, label_num):
    """
    Get stats of the instrument & shadow coordinates of one label from one lable map

    Args:
        label_map: (H, W)
        label_num: 2 or 4 in our case, 2 for instrument, 4 for shadow
    Returns:
        x_avg: average x coordinate
        y_avg: average y coordinate
        x_span: x span (height)
        y_span: y span (width)
        n_labels: number of coordinate pairs
    """
    assert len(label_map.shape)==2, f'label map must be a 2D array, but got shape {label_map.shape}'
    if isinstance(label_map, torch.Tensor):
        label_map = label_map.cpu().numpy()
    coords = np.argwhere(label_map==label_num)
    assert coords.shape[1] == len(label_map.shape), f'coords.shape[1] must equals ndim, but got shape {coords.shape}'
    x_avg = np.average(coords[:, 0])
    y_avg = np.average(coords[:, 1])
    x_span = 0
    y_span = 0
    if len(coords) > 0:
        x_span = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_span = np.max(coords[:, 1]) - np.min(coords[:, 1])
    n_label = len(coords)
    return (x_avg, y_avg, x_span, y_span, n_label)


def attribute_label_from_segmentation(seg, normalize=False):
    """
    Get attribute label from one segmentation map

    Args:
        seg: (H, W)
        normalize: if True, normalize the label to [0, 1]
    Returns:
        attribute:  (10)
            (instrument_presence,
            instrument_x_avg,
            instrument_y_avg,
            instrument_x_span,
            instrument_y_span,
            shadow_presence,
            shadow_x_avg,
            shadow_y_avg,
            shadow_x_span,
            shadow_y_span)
    """
    label = [0] * 10
    assert len(seg.shape) < 4, f"the segmentation should not be batched, got shape {seg.shape}"
    if len(seg.shape) >= 3:
        # assert seg.shape[0] == 5, f"illegal segmentation shape: {seg.shape}"
        seg = seg.argmax(dim=0)
    x2, y2, h2, w2, n2 = get_coor_avg(seg, 2) #instrument
    x4, y4, h4, w4, n4 = get_coor_avg(seg, 4) #shadow
    if n2 > 10:
        label[0] = 1
        label[1] = x2
        label[2] = y2
        label[3] = h2
        label[4] = w2
    if n4 > 10:
        label[5] = 1
        label[6] = x4
        label[7] = y4
        label[8] = h4
        label[9] = w4
    if normalize:
        label[1] /= 1024
        label[2] /= 512
        label[3] /= 1024
        label[4] /= 512
        label[6] /= 1024
        label[7] /= 512
        label[8] /= 1024
        label[9] /= 512
    return label


def get_train_loader_from_checkpoint(model_path):
    """
    Get train dataloader from the Coach instance checkpoint
    """
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    optss = Namespace(**opts)
    optss.batch_size = 8
    optss.stylegan_weights = model_path
    optss.load_partial_weights = True

    coach = Coach(optss)
    train_loader = coach.train_dataloader
    return train_loader