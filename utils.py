import torch
import numpy as np
from scipy.stats import lognorm
from einops import repeat
from torch import Tensor
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from typing import Callable


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


    # timesteps = torch.tensor(get_flux_schedule(
    #     num_steps=num_steps, 
    #     image_seq_len=(pl_module.max_depth*pl_module.res*pl_module.res)//(4**3), 
    #     shift=True
    # ), device=pl_module.device).unsqueeze(-1)

def get_ids(height: int, width: int, cls_token_len: int):
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

def logit_normal_sample(mu: float, sigma: float, num_samples: int, epsilon: float = 1e-3):
    s = sigma
    scale = np.exp(mu)
    dist = lognorm(s=s, scale=scale)
    
    x = np.linspace(0.001, 0.999, num_samples)
    pdf = dist.pdf(x)
    # Enforce a minimum non-zero probability for small x values
    pdf = np.maximum(pdf, epsilon)
    pdf = pdf / pdf.sum()
    
    return torch.tensor(pdf, dtype=torch.float32), torch.tensor(x, dtype=torch.float32)

@torch.no_grad()
def euler_sampler(model: nn.Module, y: Tensor, z1: Tensor, 
                  x_ids: Tensor, y_ids: Tensor,
                  steps: int=24):
    b, l, d = y.shape

    timesteps = model.sample_timesteps(steps).to(model.dtype)
    timesteps = timesteps.sort(descending=True).values

    # noise
    zt = z1.clone()
    for i, t in enumerate(tqdm(timesteps[:-1])):
        t_next = timesteps[i+1]
        assert t_next < t, f"t_next {t_next} must be less than t {t}"

        t = t.repeat(b)
        v_theta = model.forward(x=zt, x_ids=x_ids, y=y, y_ids=y_ids, timesteps=t)

        t_next = t_next.repeat(b, 1, 1)
        t = t.reshape(b, 1, 1)

        zt = zt + (t_next - t) * v_theta
    return zt


@torch.no_grad()
def euler_sampler_test(model: nn.Module, y: Tensor, z1: Tensor, z0: Tensor,
                  x_ids: Tensor, y_ids: Tensor,
                  steps: int=24):
    b, l, d = y.shape


    timesteps = torch.tensor(get_flux_schedule(
        num_steps=steps, 
        image_seq_len=l, 
        shift=True
    ), device=model.device).unsqueeze(-1)

    # timesteps = model.sample_timesteps(steps).to(model.dtype)
    # timesteps = timesteps.sort(descending=True).values
    print(f'Timesteps: {timesteps}')

    plt.plot(np.arange(len(timesteps)), timesteps.cpu().numpy(), 'o-')
    plt.grid(True, alpha=0.3)
    plt.savefig('timesteps.png', dpi=300)
    plt.close("all")

    v = z1 - z0
    losses = []
    # noise
    zt = z1.clone()
    for i, t in enumerate(timesteps[:-1]):
        t_next = timesteps[i+1]
        assert t_next < t, f"t_next {t_next} must be less than t {t}"

        t = t.repeat(b)
        v_theta = model.forward(x=zt, x_ids=x_ids, y=y, y_ids=y_ids, timesteps=t)

        loss = F.mse_loss(v_theta, v)
        losses.append(loss.item())

        t_next = t_next.repeat(b, 1, 1)
        t = t.reshape(b, 1, 1)

        zt = zt + (t_next - t) * v_theta

        print(f'Delta t: t_next:{t_next.flatten().item()} - t:{t.flatten().item()} := {(t_next - t).flatten().item()}. Loss: {loss.item()}')
    
    plt.plot(np.arange(len(losses)), losses, 'o-')
    plt.grid(True, alpha=0.3)
    plt.savefig('losses.png', dpi=300)
    plt.close("all")
    # import pdb; pdb.set_trace()

    return zt
    

def test_logit_normal_sample():
    import matplotlib.pyplot as plt
    pdf, x = logit_normal_sample(1, 1, 10000)
    samples = np.random.choice(x.numpy(), size=100000, p=pdf.numpy())
    print(samples.shape, f'Range: {samples.min()} - {samples.max()}')
    plt.hist(samples, bins=100, density=True, alpha=0.7, label='Sampled Points')
    plt.plot(x, pdf*10000, 'r-', linewidth=2, label='Theoretical PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Sampled vs Theoretical Logit Normal Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    test_logit_normal_sample()