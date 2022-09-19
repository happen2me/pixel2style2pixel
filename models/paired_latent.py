import torch
import pytorch_lightning as pl
from models.latent2latent import Latent2Latent
from utils.pair_utils import train_pairs_batch


class LightningPairedLatent2Latent(pl.LightningModule):
    """
    Lightning module for paired latent2latent.
    """
    def __init__(self, style_model):
        super().__init__()
        self.style_model = style_model
        self.latent_model = Latent2Latent()
        # Freeze stylegan
        for p in self.style_model.parameters():
            p.requires_grad = False

    def forward(self, x, delta_attribute):
        return self.latent_model(x, delta_attribute)

    def training_step(self, batch, batch_idx):
        src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr = batch
        return train_pairs_batch(src_seg, src_bscan, src_attr, dst_seg, dst_bscan, dst_attr,
                                 self.style_model, self, self.device)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.latent_model.parameters(), lr=0.01)


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
