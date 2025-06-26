import math
from typing import Callable
from einops import repeat
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np
import matplotlib.pyplot as plt



def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_flux_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def get_ids_2dlatent_1dcond(height: int, width: int, cls_token_len: int):
    # condition is a sequence of cls_token_len tokens
    ids_cond = torch.zeros(cls_token_len, 3)
    ids_cond[..., 0] = torch.arange(cls_token_len)
    ids_cond = repeat(ids_cond, "l c -> b l c", b=1)

    H, W = height, width
    ids_x = torch.zeros(H, W, 3)
    ids_x[..., 1] = torch.arange(H)[:, None]
    ids_x[..., 2] = torch.arange(W)[None, :]
    ids_x = repeat(ids_x, "h w c -> b (h w) c", b=1)
    return ids_x, ids_cond


def sample_timesteps(batch_size, device, eps=1e-4) -> Tensor:
    t = torch.distributions.Beta(concentration1=2.5, concentration0=2.0).sample((batch_size,)).to(device)
    t = t.clamp(eps, 1.0 - eps)                        # (0, 1) guard-band
    assert torch.all((t >= 0.0) & (t <= 1.0)), "t out of [0,1] range"
    return t


@torch.no_grad()
def euler_step(model: nn.Module, y: Tensor, z1: Tensor, z0: Tensor,
                  x_ids: Tensor, y_ids: Tensor,
                  steps: int=24):
    b, l, d = y.shape

    timesteps = torch.tensor(get_flux_schedule(
        num_steps=steps, 
        image_seq_len=l, 
        shift=True
    ), device=model.device).unsqueeze(-1)

    v = z1 - z0 if z0 is not None else None

    losses = []
    # noise
    zt = z1.clone()
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i+1]
        assert t_next < t, f"t_next {t_next} must be less than t {t}"

        t = t.repeat(b)
        v_theta = model.forward(x=zt, x_ids=x_ids, y=y, y_ids=y_ids, timesteps=t)

        if v is not None:
            loss = F.mse_loss(v_theta, v)
            losses.append(loss.item())

        t_next = t_next.repeat(b, 1, 1)
        t = t.reshape(b, 1, 1)

        zt = zt + (t_next - t) * v_theta
    
    return zt, losses


@torch.no_grad()
def sample_test(model:nn.Module, cond_encoder:nn.Module,
                cond: Tensor, target: Tensor, autoencoder:nn.Module,
                steps: int=24) -> Tensor:
    b, c, h, w = cond.shape
    assert h == w == 256, f"Condition must be 256x256, got {h}x{w}"
    assert c == 3, f"Condition must have 3 channels, got {c}"
    
    y = cond_encoder(cond.to(model.device))
    y = y.view(b, 512, -1).permute(0, 2, 1).contiguous()
    x_ids, y_ids = get_ids_2dlatent_1dcond(32, 32, y.shape[1])
    x_ids = repeat(x_ids, 'a l c -> (a b) l c', b=b).to(model.device)
    y_ids = repeat(y_ids, 'a l c -> (a b) l c', b=b).to(model.device)

    if target is not None:
        z0 = autoencoder.encode(target)
        z0 = z0.view(b, 4, -1).permute(0, 2, 1).contiguous()
    else:
        z0 = None

    z1 = torch.randn((b, 32*32, 4), device=model.device, dtype=model.dtype)
    zt = euler_step(model, y, z1, z0, x_ids, y_ids, steps)

    print(f'Final Loss: {F.mse_loss(zt, z0)}')

    if target is not None:
        z0 = z0.permute(0, 2, 1).view(b, 4, 32, 32).contiguous()
        z0 = autoencoder.decode(z0)
        decoded_input = z0.clamp(0, 1).squeeze(0)
    else:
        decoded_input = None

    zt = zt.permute(0, 2, 1).view(b, 4, 32, 32).contiguous()
    zt = autoencoder.decode(zt)

    decoded_img = zt.clamp(0, 1).squeeze(0)
    return decoded_img, decoded_input