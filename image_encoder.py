import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        out_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h



# class ImageEncoder(nn.Module):
    # def __init__(self, in_channels=3, hidden_channels=[32, 64, 256, 512], out_channels=1024):
    #     super().__init__()
    #     assert len(hidden_channels) == 3, f"Expected 3 hidden channels, got {len(hidden_channels)}"
        
    #     # Initial projection
    #     self.conv_in = nn.Conv2d(in_channels, hidden_channels[0], 3, padding=1)
        
    #     # 8x downsampling: 2x2x2 = 8
    #     self.downs = nn.ModuleList([
    #         nn.Sequential(
    #             ResBlock(hidden_channels[0]),
    #             nn.Conv2d(hidden_channels[0], hidden_channels[1], 4, stride=2, padding=1)
    #         ),
    #         nn.Sequential(
    #             ResBlock(hidden_channels[1]),
    #             nn.Conv2d(hidden_channels[1], hidden_channels[2], 4, stride=2, padding=1)
    #         ),
    #         nn.Sequential(
    #             ResBlock(hidden_channels[2]),
    #             nn.Conv2d(hidden_channels[2], out_channels, 4, stride=2, padding=1)
    #         )
    #     ])
        
    #     self.final_norm = nn.GroupNorm(1, out_channels)
        
    # def forward(self, x):
    #     h = self.conv_in(x)
        
    #     for down in self.downs:
    #         h = down(h)
            
    #     h = self.final_norm(h)
    #     # More efficient reshaping to tokens: (B, C, H, W) -> (B, H*W, C)
    #     B, C = h.shape[:2]
    #     tokens = h.view(B, C, -1).permute(0, 2, 1).contiguous()
    #     return tokens

