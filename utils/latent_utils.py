import random
import numpy as np
import torch
import torch.nn.functional as F
from utils.regressor_utils import attribute_label_from_segmentation

def modify_attribute(attribute, correlation_matrix, change_nr=1, change=None):
    """
    Modify the attribute by 1 dimension.

    Args:
        attribute: (batch_size, 10)
        correlation_matrix: (10, 10) correlation matrix of all attributes
        change_nr: number of dimensions to change
        change: a scala in range (-0.5, 0.5). If None, a random scala will be generated.
    Returns:
        modified attribute: (batch_size, 10)
        actual_change: (batch_size, 10) the corrected actual change of each dimension
    """
    if change is None:
        change = random.random() - 0.5
    attribute = np.array(attribute)
    curr_val = attribute[change_nr]
    changed_val = np.clip(curr_val + change, 0, 1)
    actual_change = changed_val - curr_val
    changed_attribute = np.clip(attribute + actual_change * correlation_matrix[change_nr], 0, 1)
    return changed_attribute, actual_change

def get_latent(style_model, seg, device):
    """
    Get the latent of a given segmentation map with the style model
    Returned tensors are detached and moved to cpu
    
    Args:
        style_model: the style model
        seg: a single or a batch of segmentation map
        device: device to run the model
    Returns:
        pred: (n_batch, 18, 512)
        latent: (n_batch, 18, 512)
        codes: same as latent (for debug purpose)
    """
    style_model = style_model.to(device)
    if len(seg.size()) < 4:
        seg = seg.unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred, latent, codes = style_model(seg, return_latents=True, return_codes=True)
    return pred.detach().cpu(), latent.detach().cpu(), codes.detach().cpu()


def train_imgs_batch(segs, style_model, regressor, latent_model, device, correlation_matrix):
    """
    Train the latent to latent model on a batch of images.

    Args:
        segs: (n_batch, 5, 256, 256)
        style_model: the style model (frozen)
        regressor: the regressor (frozen)
        latent_model: the latent to latent model
        device: device to run the model
        correlation_matrix: (10, 10) correlation matrix of all attributes
    Returns:
        loss: the loss of the batch
    """
    with torch.no_grad():
        _, _, w_latents = get_latent(style_model, segs, device)
    w_latents = w_latents.to(device)
    attributes = [attribute_label_from_segmentation(seg, normalize=True) for seg in segs]
    modified = [modify_attribute(attribute, correlation_matrix, change_nr=1) for attribute in attributes]
    modified_scale = [item[1] for item in modified]
    attributes = torch.Tensor(attributes).to(device)
    attributes_mask_1 = attributes[:, 0].unsqueeze(1).repeat(1, 5)
    attributes_mask_2 = attributes[:, 5].unsqueeze(1).repeat(1, 5)
    attributes_mask = torch.cat([attributes_mask_1, attributes_mask_2], dim=-1)
    modified_attributes = np.array([item[0] for item in modified])
    modified_attributes = torch.Tensor(modified_attributes).to(device)
    delta_attributes = modified_attributes - attributes

    w_n =  latent_model(w_latents, delta_attributes)
    w_n = 0.7*w_latents + 0.3*w_n

    # cycle loss
    w_n_c = latent_model(w_n, -delta_attributes)
    w_n_c = 0.7*w_latents+0.3*w_n_c
    loss_cycle = F.l1_loss(w_n_c, w_latents)
    
    # Identity loss
    w_i = latent_model(w_latents, 0*delta_attributes)
    w_i = 0.7*w_latents + 0.3*w_i
    loss_identity = F.l1_loss(w_i, w_latents)
    
    # neighborhood loss
    loss_neighborhood = F.mse_loss(w_n, w_latents)
    
    # attribute loss
    # w_n = w_n.to(device)
    generated_images = style_model(w_n, input_code=True)
    attribute_scores = regressor(generated_images)
    loss_attributes = F.mse_loss(attribute_scores * attributes_mask, modified_attributes * attributes_mask)
    
    loss_all =  loss_cycle*0.5 + loss_attributes + loss_identity*0.5 + loss_neighborhood*0.5
    return loss_all