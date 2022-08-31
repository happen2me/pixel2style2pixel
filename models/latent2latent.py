import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stylegan2.model import EqualLinear, PixelNorm
import pytorch_lightning as pl
from utils.latent_utils import train_imgs_batch

class Mapper(nn.Module):
    def __init__(self,latent_dim=512, attribute_dim=10):
        super(Mapper,self).__init__()
        layers = []
        layers.append(
            nn.Sequential(
                EqualLinear(latent_dim + attribute_dim, latent_dim, lr_mul=0.01),
                nn.Tanh(),
            )
        )
        layers.append(
            nn.Sequential(
                EqualLinear(latent_dim, latent_dim, lr_mul=0.01),
                nn.Tanh(),
            )
        )
        layers.append(
            nn.Sequential(
                EqualLinear(latent_dim, latent_dim, lr_mul=0.01)
            )
        )
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x

class Latent2Latent(nn.Module):
    def __init__(self):
        super(Latent2Latent,self).__init__()
        self.course_mapping = Mapper()
        self.medium_mapping = Mapper()
        self.fine_mapping = Mapper()

    def forward(self, x, delta_attribute):
        # delta_attribute: (batch_size, 10)
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]
        # repeat delta_attribute and concat it to x_coarse, x_medium, x_fine along the last dimension
        attribute_coarse = delta_attribute.unsqueeze(1).expand(-1, x_coarse.size(1), -1)
        attribute_medium = delta_attribute.unsqueeze(1).expand(-1, x_medium.size(1), -1)
        attribute_fine = delta_attribute.unsqueeze(1).expand(-1, x_fine.size(1), -1)
        # concat x and attribute
        x_coarse = torch.cat([x_coarse, attribute_coarse], dim=-1)
        x_medium = torch.cat([x_medium, attribute_medium], dim=-1)
        x_fine = torch.cat([x_fine, attribute_fine], dim=-1)
        x_coarse = self.course_mapping(x_coarse)
        x_medium = self.medium_mapping(x_medium)
        x_fine = self.fine_mapping(x_fine)
        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return out


class LightningLatent2Latent(pl.LightningModule):
    def __init__(self, style_model, regressor, correlation_matrix):
        super().__init__()
        self.latent2latent = Latent2Latent()
        self.style_model = style_model
        self.regressor = regressor
        self.correlation_matrix = correlation_matrix
        
        # Freeze stylegan and regressor
        for p in self.style_model.parameters():
            p.requires_grad = False
        for p in self.regressor.parameters():
            p.requires_grad = False
            
    def load_latent_state(self, latent2latent_path):
        self.latent2latent.load_state_dict(torch.load(latent2latent_path))
    
    def forward(self, x, delta_attribute):
        return self.latent2latent(x, delta_attribute)
    
    def training_step(self, batch, batch_idx):
        segs, bscans = batch
        segs = segs.to(self.device).float()
        return train_imgs_batch(segs, self.style_model, self.regressor, self, self.device, self.correlation_matrix)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.latent2latent.parameters(), lr=0.01)

if __name__ == '__main__':
    model = Latent2Latent()
    x = torch.randn(2, 9,512)
    delta_attribute = torch.randn(2,10)
    out = model(x, delta_attribute)
    print(out.size())