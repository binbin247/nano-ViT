"""ViT model configurations from Table 1 of the original ViT paper.

The paper defines three backbone sizes and five named variants:
ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, and ViT-H/14.

The class count and image size are task-dependent.  The defaults below use
224x224 RGB images and 1000 ImageNet-1K classes.
"""

from model import ViTConfig


def _make_config(
    *,
    patch_size: int,
    n_layer: int,
    ebd_dim: int,
    ffn_dim: int,
    n_head: int,
    image_size: tuple[int, int] = (224, 224),
    in_channels: int = 3,
    cls_num: int = 1000,
    dropout: float = 0.1,
) -> ViTConfig:
    """Build a project ``ViTConfig`` from an original ViT specification."""
    if ebd_dim % n_head != 0:
        raise ValueError("ebd_dim must be divisible by n_head")

    return ViTConfig(
        image_size=image_size,
        in_channels=in_channels,
        ebd_dim=ebd_dim,
        n_head=n_head,
        head_dim=ebd_dim // n_head,
        patch_size=patch_size,
        n_layer=n_layer,
        ffn_dim=ffn_dim,
        cls_num=cls_num,
        dropout_img_ebd=dropout,
        dropout_att_weight=0.0,
        dropout_att_out=dropout,
        dropout_ffn_out=dropout,
        ffn_bias=False,
    )


# The original paper's model variants.
vit_b_16 = _make_config(
    patch_size=16, n_layer=12, ebd_dim=768, ffn_dim=3072, n_head=12,
)
vit_b_32 = _make_config(
    patch_size=32, n_layer=12, ebd_dim=768, ffn_dim=3072, n_head=12,
)
vit_l_16 = _make_config(
    patch_size=16, n_layer=24, ebd_dim=1024, ffn_dim=4096, n_head=16,
)
vit_l_32 = _make_config(
    patch_size=32, n_layer=24, ebd_dim=1024, ffn_dim=4096, n_head=16,
)
vit_h_14 = _make_config(
    patch_size=14, n_layer=32, ebd_dim=1280, ffn_dim=5120, n_head=16,
)


VIT_CONFIGS: dict[str, ViTConfig] = {
    "B/16": vit_b_16,
    "B/32": vit_b_32,
    "L/16": vit_l_16,
    "L/32": vit_l_32,
    "H/14": vit_h_14,
}


__all__ = [
    "VIT_CONFIGS",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]
