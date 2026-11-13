import argparse

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ImageEmbed(nn.Module):
    """Embed image patches, add positional embeddings, and prepend a CLS token.

    Input: (batch_size, n_channels, height, width).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        height = width = args.image_size
        self.conv = nn.Conv2d(  # Project image patches into the embedding space.
            in_channels = args.in_channels,
            out_channels = args.ebd_dim,
            kernel_size = (args.patch_size, args.patch_size),
            stride = (args.patch_size, args.patch_size),
            bias = True,
        )

        assert height % args.patch_size == 0 and width % args.patch_size == 0
        seq_length = (height // args.patch_size) * (width // args.patch_size)

        self.pos_ebd = nn.Parameter(
            torch.zeros(1, seq_length+1, args.ebd_dim),
            requires_grad=True
        )
        self.cls_token = nn.Parameter(torch.zeros(1,1,args.ebd_dim), requires_grad = True)
        self.dropout = nn.Dropout(args.dropout_img_ebd)


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
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.Wq = nn.Linear(args.ebd_dim, args.n_head * args.head_dim, bias = True)
        self.Wk = nn.Linear(args.ebd_dim, args.n_head * args.head_dim, bias = True)
        self.Wv = nn.Linear(args.ebd_dim, args.n_head * args.head_dim, bias = True)
        self.output_proj = nn.Linear(args.n_head * args.head_dim, args.ebd_dim, bias = True)
        self.dropout = nn.Dropout(args.dropout_att_out)

    def forward(self, x: Tensor) -> Tensor:    # x.shape = (B = batch_size, T = seq_length + 1, C = ebd_dim)
        assert x.dim() == 3
        B, T, D = x.shape
        q = self.Wq(x).reshape(B, T, self.args.n_head, self.args.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)
        k = self.Wk(x).reshape(B, T, self.args.n_head, self.args.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)
        v = self.Wv(x).reshape(B, T, self.args.n_head, self.args.head_dim).transpose(1,2)  # (B, n_head, T, head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False,
            dropout_p = self.args.dropout_att_weight if self.training else 0.0,
        )  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1) # (B, T, n_head*head_dim)
        out = self.output_proj(out) # (B, T, D)
        return self.dropout(out)
    
class FFN(nn.Module):
    """Apply a configurable GELU MLP or SwiGLU FFN to each token.

    Input: (batch_size, seq_length + 1, ebd_dim).
    Output: (batch_size, seq_length + 1, ebd_dim).
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.ffn_type = args.ffn_type.lower()
        if self.ffn_type not in {"gelu", "swiglu"}:
            raise ValueError(
                f"Unsupported ffn_type={args.ffn_type!r}; "
                "expected 'gelu' or 'swiglu'."
            )
        if self.ffn_type == "swiglu":
            self.w1 = nn.Linear(args.ebd_dim, args.ffn_dim, bias = args.ffn_bias)
            self.w2 = nn.Linear(args.ebd_dim, args.ffn_dim, bias = args.ffn_bias)
            self.w3 = nn.Linear(args.ffn_dim, args.ebd_dim, bias = args.ffn_bias)
            self.dropout_out = nn.Dropout(args.dropout_ffn_out)

        if self.ffn_type == "gelu":
            self.w1 = nn.Linear(args.ebd_dim, args.ffn_dim, bias = args.ffn_bias)
            self.w2 = nn.Linear(args.ffn_dim, args.ebd_dim, bias = args.ffn_bias)
            self.dropout_inner = nn.Dropout(args.dropout_ffn_inner)
            self.dropout_out = nn.Dropout(args.dropout_ffn_out)

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

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.ln_mha = nn.LayerNorm(args.ebd_dim, eps=1e-6)
        self.mha = MultiHeadSelfAttention(args)
        self.ln_ffn = nn.LayerNorm(args.ebd_dim, eps=1e-6)
        self.ffn = FFN(args)

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
    Output: (batch_size, num_classes).
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.w1 = nn.Linear(args.ebd_dim, args.ebd_dim)
        self.out_proj = nn.Linear(args.ebd_dim, args.num_classes)

    def forward(self, x: Tensor) -> Tensor: # (batch_size, seq_length + 1, ebd_dim)
        x = x[:, 0, :]  # get cls token (batch_size, ebd_dim)
        return self.out_proj(torch.tanh(self.w1(x))) # (batch_size, num_classes)



class ViT(nn.Module):
    """Classify images with patch embedding, Transformer blocks, and a CLS head.

    Input: (batch_size, n_channels, height, width).
    Output: class logits with shape (batch_size, num_classes).
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.img_ebd = ImageEmbed(args)
        self.encoder = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layer)])
        self.ln_cls_head = nn.LayerNorm(args.ebd_dim, eps=1e-6)
        self.cls_head = ClassifierHead(args)

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
