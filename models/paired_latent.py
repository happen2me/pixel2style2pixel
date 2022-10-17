import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.latent2latent import Latent2Latent
from utils.pair_utils import train_pairs_batch
import transformers


class LinearLatent2Latent(nn.Module):
    '''
    A MLP that transform latent to a same-dimension latent
    '''
    def __init__(self, latent_dim=512, attribute_dim=10):
        super(LinearLatent2Latent, self).__init__()
        self.latent_dim = latent_dim
        self.attribute_dim = attribute_dim
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim + attribute_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, delta_attribute):
        # delta_attribute: (batch_size, 10)
        x = torch.cat([x, delta_attribute], dim=-1)
        x = self.mapping(x)
        return x

class SelfAttentionLatent2Latent(nn.Module):
    '''
    A multihead attention based latent2latent model
    '''
    def __init__(self, latent_dim=512, attribute_dim=10):
        super(SelfAttentionLatent2Latent, self).__init__()
        self.latent_dim = latent_dim
        self.attribute_dim = attribute_dim
        self.mha1 = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, dropout=0.1,
            kdim=attribute_dim, vdim=latent_dim, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, dropout=0.1, 
            batch_first=True)
    
    def forward(self, x, delta_attribute):
        # delta_attribute: (batch_size, 10)
        delta_attribute = delta_attribute.unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.mha1(x, delta_attribute, x)[0]
        x = self.mha2(x, x, x)[0]
        return x

class LightningPairedLatent2Latent(pl.LightningModule):
    """
    Lightning module for paired latent2latent.
    """
    def __init__(self, style_model, lr=0.01):
        super().__init__()
        self.style_model = style_model
        self.latent_model = Latent2Latent()
        # Freeze stylegan
        for p in self.style_model.parameters():
            p.requires_grad = False
        self.lr = lr

    def forward(self, x, delta_attribute):
        return self.latent_model(x, delta_attribute)

    def training_step(self, batch, batch_idx):
        src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr = batch
        return train_pairs_batch(src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr,
                                 self.style_model, self, self.device)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.latent_model.parameters(), lr=(self.lr or self.learning_rate))


if __name__ == '__main__':
    # Test LightningPairedLatent2Latent.
    import pickle
    from torch.utils.data import DataLoader
    from pytorch_lightning import Trainer
    from utils.pair_utils import prepare_paired_dataset

    dataset = prepare_paired_dataset()
    dataloader = DataLoader(dataset, batch_size=4)
    trainer = Trainer(fast_dev_run=True, accelerator='gpu', devices=1)
    style_model = pickle.load(open('artifacts/cache/debug/models/style_model.pkl', 'rb'))
    model = LightningPairedLatent2Latent(style_model)
    trainer.fit(model, dataloader)
