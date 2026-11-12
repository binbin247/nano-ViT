import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ViTConfig:
    ebd_dim: int
    n_head: int
    head_dim: int
    patch_size: int
    n_layer: int
    ffn_dim: int
    dropout_att_weight: float = 0.0
    dropout_att_out: float = 0.0
    dropout_ffn_out: float = 0.0
    ffn_bias: bool = False

class ImageEmbed(nn.Module):
    """Embed image patches, add positional embeddings, and prepend a CLS token.

    Input: (batch_size, n_channels, width, height).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(
        self,
        image_size: tuple[int, int, int], # n_channels, width, height
        cfg: ViTConfig,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(  # Project image patches into the embedding space.
            in_channels = image_size[0],
            out_channels = cfg.ebd_dim,
            kernel_size = (cfg.patch_size, cfg.patch_size),
            stride = (cfg.patch_size, cfg.patch_size),
            bias = True,
        )

        assert image_size[1] % cfg.patch_size == 0 and image_size[2] % cfg.patch_size == 0 
        seq_length = (image_size[1] // cfg.patch_size) * (image_size[2] // cfg.patch_size)

        self.pos_ebd = nn.Parameter(
            torch.zeros(1, seq_length, cfg.ebd_dim),
            requires_grad=True
        )

        self.cls_token = nn.Parameter(torch.zeros(1,1,cfg.ebd_dim), requires_grad = True)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor)-> Tensor:
        x = self.conv(x)                                                     # (B, ebd_dim, width//patch_size, height//patch_size)
        x = x.reshape(x.size(0), x.size(1), -1)                              # (B, ebd_dim, seq_length)
        x = x.transpose(-1, -2)                                              # (B, seq_length, ebd_dim)
        x += self.pos_ebd
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)  # (B, seq_length + 1, ebd_dim)
        return self.dropout(x)

    
class MultiHeadSelfAttention(nn.Module):
    """Compute multi-head self-attention over CLS and patch tokens.

    All tokens can attend to each other; no causal mask is used.
    Input: (batch_size, seq_length + 1, ebd_dim).
    Output: (batch_size, seq_length + 1, ebd_dim).
    seq_length counts patches; CLS occupies index 0.
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.Wq = nn.Linear(cfg.ebd_dim, cfg.n_head * cfg.head_dim, bias = True)
        self.Wk = nn.Linear(cfg.ebd_dim, cfg.n_head * cfg.head_dim, bias = True)
        self.Wv = nn.Linear(cfg.ebd_dim, cfg.n_head * cfg.head_dim, bias = True)
        self.output_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.ebd_dim, bias = True)
        self.dropout = nn.Dropout(cfg.dropout_att_out)

    def forward(self, x: Tensor):    # x.shape = (B = batch_size, T = seq_length + 1, C = ebd_dim)
        assert x.dim() == 3
        B, T, D = x.shape
        q = self.Wq(x).reshape(B, T, self.cfg.n_head, self.cfg.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)
        k = self.Wk(x).reshape(B, T, self.cfg.n_head, self.cfg.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)
        v = self.Wv(x).reshape(B, T, self.cfg.n_head, self.cfg.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False,
            dropout_p = self.cfg.dropout_att_weight if self.training else 0.0, 
        )  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1) # (B, T, n_head*head_dim)
        out = self.output_proj(out) # (B, T, D)
        return self.dropout(out)
    
class FFN(nn.Module):
    """
    # TODO: add comment.
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Linear(cfg.ebd_dim, cfg.ffn_dim, bias = cfg.ffn_bias)
        self.w2 = nn.Linear(cfg.ebd_dim, cfg.ffn_dim, bias = cfg.ffn_bias)
        self.w3 = nn.Linear(cfg.ffn_dim, cfg.ebd_dim, bias = cfg.ffn_bias)
        self.dropout_out = nn.Dropout(cfg.dropout_ffn_out)

    def forward(self, x: Tensor):    # x.shape = (B = batch_size, T = seq_length+1, C = ebd_dim)
        return self.dropout_out(self.w3(F.silu(self.w1(x)) * self.w2(x)))
