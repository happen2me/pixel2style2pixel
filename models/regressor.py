import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.regressor_utils import attribute_label_from_segmentation

# Expected input size is (256, 256)
class Regressor(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        convs = []
        in_channels=1
        for _ in range(5):            
            convs.append(nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1))
            in_channels = in_channels * 2
            convs.append(nn.LeakyReLU())
            convs.append(nn.MaxPool2d(2, 2))
        self.convs = nn.Sequential(*convs)
        self.fc1 = nn.Sequential(
            nn.Linear(8*8*32, 256),
            nn.LeakyReLU()
        )
        self.fc2 =  nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Linear(64, output_size)
        

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LightningRegressor(pl.LightningModule):
    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor
    
    def training_step(self, batch, batch_idx):
        seg, bscan = batch
        attributes = [attribute_label_from_segmentation(s) for s in seg]
        attributes = torch.Tensor(attributes).float().cuda()
        # Normalize by width and heights
        attributes[:, 1] /= 1024
        attributes[:, 2] /= 512
        attributes[:, 3] /= 1024
        attributes[:, 4] /= 512
        attributes[:, 6] /= 1024
        attributes[:, 7] /= 512
        attributes[:, 8] /= 1024
        attributes[:, 9] /= 512
        mask1 = attributes[:, 0].unsqueeze(1).repeat(1, 5)
        mask2 = attributes[:, 5].unsqueeze(1).repeat(1, 5)
        masks = torch.cat((mask1, mask2), dim=1)
        assert attributes.shape==masks.shape, f"shape mismatch {attributes.shape} and {masks.shape}"
        pred = self.regressor(bscan)
        loss = F.mse_loss(pred*masks, attributes*masks)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        