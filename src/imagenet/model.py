import torch
import torch.nn as nn
from .unet import UNetModel

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=1001, in_channels=3, model_channels=128, out_channels=3,
                 num_res_blocks=2, channel_mult=(1, 2, 4, 8), attention_resolutions=(16, 8), dropout=0.0):
        super().__init__()
        self.model = UNetModel(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x_t, t, y):
        return self.model(x_t, t, y)
