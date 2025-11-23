import torch
import torch.nn as nn
from .unet import UNetModel

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    def __init__(self, x_dim=2, y_num_classes=4, y_emb_dim=16, hidden_dim=256):
        super().__init__()
        self.y_embedding = nn.Embedding(num_embeddings=y_num_classes, embedding_dim=y_emb_dim)
        
        input_dim = x_dim + 1 + y_emb_dim  # x_t (2) + t (1) + y_emb (16)
        output_dim = x_dim  # u_t same dimension as x_t

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
            )
    

    def forward(self, x_t, t, y):
        y_emb = self.y_embedding(y)               # (batch_size, y_emb_dim)
        inp = torch.cat([x_t, t, y_emb], dim=-1)
        return self.net(inp)

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3, model_channels=128, out_channels=3,
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



