import torch
import numpy as np
from scipy.stats import lognorm
from einops import repeat
from torch import Tensor
from tqdm import tqdm
from torch import nn

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

def logit_normal_sample(mu: float, sigma: float, num_samples: int):
    s = sigma
    scale = np.exp(mu)
    dist = lognorm(s=s, scale=scale)
    
    x = np.linspace(0.001, 0.999, num_samples)
    pdf = dist.pdf(x)
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