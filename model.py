import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ViTConfig:
    image_size: tuple[int, int]  # height, width
    in_channels: int
    ebd_dim: int
    n_head: int
    head_dim: int
    patch_size: int
    n_layer: int
    ffn_dim: int
    cls_num: int
    ffn_type: str = "gelu"
    dropout_img_ebd: float = 0.0
    dropout_att_weight: float = 0.0
    dropout_att_out: float = 0.0
    dropout_ffn_inner: float = 0.0
    dropout_ffn_out: float = 0.0
    ffn_bias: bool = False

class ImageEmbed(nn.Module):
    """Embed image patches, add positional embeddings, and prepend a CLS token.

    Input: (batch_size, n_channels, height, width).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        height, width = cfg.image_size
        self.conv = nn.Conv2d(  # Project image patches into the embedding space.
            in_channels = cfg.in_channels,
            out_channels = cfg.ebd_dim,
            kernel_size = (cfg.patch_size, cfg.patch_size),
            stride = (cfg.patch_size, cfg.patch_size),
            bias = True,
        )

        assert height % cfg.patch_size == 0 and width % cfg.patch_size == 0
        seq_length = (height // cfg.patch_size) * (width // cfg.patch_size)

        self.pos_ebd = nn.Parameter(
            torch.zeros(1, seq_length+1, cfg.ebd_dim),
            requires_grad=True
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,cfg.ebd_dim), requires_grad = True)
        self.dropout = nn.Dropout(cfg.dropout_img_ebd)


    def forward(self, x: Tensor)-> Tensor:
        x = self.conv(x)                                                     # (B, ebd_dim, height//patch_size, width//patch_size)
        x = x.reshape(x.size(0), x.size(1), -1)                              # (B, ebd_dim, seq_length)
        x = x.transpose(-1, -2)                                              # (B, seq_length, ebd_dim)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)  # (B, seq_length + 1, ebd_dim)
        x += self.pos_ebd
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

    def forward(self, x: Tensor) -> Tensor:    # x.shape = (B = batch_size, T = seq_length + 1, C = ebd_dim)
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
    """Apply a configurable GELU MLP or SwiGLU FFN to each token.

    Input: (batch_size, seq_length + 1, ebd_dim).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.ffn_type = cfg.ffn_type.lower()
        if self.ffn_type not in {"gelu", "swiglu"}:
            raise ValueError(
                f"Unsupported ffn_type={cfg.ffn_type!r}; "
                "expected 'gelu' or 'swiglu'."
            )
        if self.ffn_type == "swiglu":
            self.w1 = nn.Linear(cfg.ebd_dim, cfg.ffn_dim, bias = cfg.ffn_bias)
            self.w2 = nn.Linear(cfg.ebd_dim, cfg.ffn_dim, bias = cfg.ffn_bias)
            self.w3 = nn.Linear(cfg.ffn_dim, cfg.ebd_dim, bias = cfg.ffn_bias)
            self.dropout_out = nn.Dropout(cfg.dropout_ffn_out)

        if self.ffn_type == "gelu":
            self.w1 = nn.Linear(cfg.ebd_dim, cfg.ffn_dim, bias = cfg.ffn_bias)
            self.w2 = nn.Linear(cfg.ffn_dim, cfg.ebd_dim, bias = cfg.ffn_bias)
            self.dropout_inner = nn.Dropout(cfg.dropout_ffn_inner)
            self.dropout_out = nn.Dropout(cfg.dropout_ffn_out)

    def forward(self, x: Tensor):    # x.shape = (B = batch_size, T = seq_length+1, C = ebd_dim)
        if self.ffn_type == "swiglu":
            return self.dropout_out(self.w3(F.silu(self.w1(x)) * self.w2(x)))
        if self.ffn_type == "gelu":
            return self.dropout_out(self.w2(self.dropout_inner(F.gelu(self.w1(x)))))

class TransformerBlock(nn.Module):
    """Apply a pre-norm Transformer block with configurable FFN.

    Input: (batch_size, seq_length + 1, ebd_dim).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.ln_mha = nn.LayerNorm(cfg.ebd_dim, eps = 1e-5)
        self.mha = MultiHeadSelfAttention(cfg)
        self.ln_ffn = nn.LayerNorm(cfg.ebd_dim, eps = 1e-5)
        self.ffn = FFN(cfg)

    def forward(self, x: Tensor) -> Tensor:  # x.shape = (B, T, C)
        """
        Pre-norm transformer block.
        """
        x = self.mha(self.ln_mha(x)) + x
        x = self.ffn(self.ln_ffn(x)) + x
        return x

class ClassifierHead(nn.Module):
    """Project the CLS token representation to class logits.

    Input: (batch_size, seq_length + 1, ebd_dim).
    Output: (batch_size, cls_num).
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.ebd_dim, cfg.ebd_dim)
        self.out_proj = nn.Linear(cfg.ebd_dim, cfg.cls_num)

    def forward(self, x: Tensor) -> Tensor: # (batch_size, seq_length + 1, ebd_dim)
        x = x[:, 0, :]  # get cls token (batch_size, ebd_dim)
        return self.out_proj(torch.tanh(self.w1(x))) # (batch_size, cls_num)



class ViT(nn.Module):
    """Classify images with patch embedding, Transformer blocks, and a CLS head.

    Input: (batch_size, n_channels, height, width).
    Output: class logits with shape (batch_size, cls_num).
    """
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.img_ebd = ImageEmbed(cfg)
        self.encoder = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_cls_head = nn.LayerNorm(cfg.ebd_dim)
        self.cls_head = ClassifierHead(cfg)

        # initialize all the weights
        self.apply(self._init_weights)

    def forward(self, x:Tensor) -> Tensor:  # x input batched images (batch_size, n_channels, height, width)
        x = self.img_ebd(x)
        for block in self.encoder:
            x = block(x)
        logits = self.cls_head(self.ln_cls_head(x))
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, mean = 0, std = 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, ImageEmbed):
            nn.init.trunc_normal_(module.pos_ebd, mean=0, std=0.02)
            nn.init.trunc_normal_(module.cls_token, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
