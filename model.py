from dataclasses import dataclass

import torch
from torch import Tensor, nn
from typing import List

try:
    from layers import (
        DoubleStreamBlock,
        EmbedND,
        LastLayer,
        MLPEmbedder,
        SingleStreamBlock,
        timestep_embedding,
    )
except:
    from .layers import (
        DoubleStreamBlock,
        EmbedND,
        LastLayer,
        MLPEmbedder,
        SingleStreamBlock,
        timestep_embedding,
    )

@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    long_short_attn_freq: float # defines the frequency of the short attention blocks. (1/long_short_attn_freq) gives the time interval between long attention blocks. 0 means no short attention blocks.
    num_conds: int


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        assert self.params.long_short_attn_freq <= 1, "long_short_attn_freq must be <= 1"
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        # self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)
        # using multiple txt_ins for multiple conditions. Todo: check if this is overkill.
        self.txt_ins = [nn.Linear(params.context_in_dim, self.hidden_size) for _ in range(params.num_conds)]

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    @torch.no_grad()
    def build_attn_mask(self, txt: Tensor, img: Tensor, latent_dim: tuple[int, int, int]) -> Tensor:
        num_cond_tokens = txt.shape[1]
        num_img_tokens = img.shape[1]
        num_tokens_per_layer = num_img_tokens // latent_dim[0]

        # build grouped attention mask for concatenated [txt | img] tokens
        total_tokens = num_cond_tokens + num_img_tokens
        allowed = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=img.device)
        # cond queries attend everywhere; everyone attends to cond keys
        allowed[:num_cond_tokens, :] = True
        allowed[:, :num_cond_tokens] = True
        # grouped attention among image tokens
        for group_start in range(0, num_img_tokens, num_tokens_per_layer):
            group_end = min(group_start + num_tokens_per_layer, num_img_tokens)
            q_start = num_cond_tokens + group_start
            q_end = num_cond_tokens + group_end
            allowed[q_start:q_end, q_start:q_end] = True
        # attn_mask = ~allowed  # boolean mask: True values are masked
        return allowed

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: List[Tensor],  # txt is condition(which is orignal svg as image)
        txt_ids: List[Tensor],
        timesteps: Tensor,
        latent_dim: tuple[int, int, int],
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        if y is not None:
            vec = vec + self.vector_in(y)

        txts = [self.txt_in(txt[i]) for i, self.txt_in in enumerate(self.txt_ins)]
        txt = torch.cat(txts, dim=1)
        txt_ids = torch.cat(txt_ids, dim=1)

        assert txt.shape[1] == txt_ids.shape[1], f"txt and txt_ids must have the same number of tokens, got txt:{txt.shape} and txt_ids:{txt_ids.shape}"

        attn_mask = self.build_attn_mask(txt, img, latent_dim)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        # import pdb; pdb.set_trace()

        short_attn_freq = len(self.single_blocks) + 1 if self.params.long_short_attn_freq == 0 else int(1/self.params.long_short_attn_freq)
        for k, block in enumerate(self.single_blocks):
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask if k % short_attn_freq == 0 else None)

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


def test_model():
    params=FluxParams(
        in_channels=2,
        out_channels=2,
        vec_in_dim=4,
        context_in_dim=4,
        hidden_size=1024,
        mlp_ratio=4.0,
        num_heads=8,
        depth=2,
        depth_single_blocks=4,
        axes_dim=[16, 56, 56], # sum(axes_dim) == hidden_size // num_heads
        theta=10_000,
        qkv_bias=True,
        guidance_embed=False,
        long_short_attn_freq=0.5,
    )

    data = torch.randn(2, 1024, 2)
    cond = torch.randn(2, 1024, 4)

    data_ids = torch.randint(0, 1024, (2, 1024, 3))
    cond_ids = torch.randint(0, 1024, (2, 1024, 3))

    ts = torch.randn(2)
    y = torch.randn(2, 4)

    model = Flux(params)
    out = model(data, data_ids, cond, cond_ids, ts, y)
    print(out.shape)


if __name__ == "__main__":
    test_model()