import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops import repeat

from model import Flux, FluxParams
from image_encoder import ImageEncoder
from autoencoder import HuggingFaceAutoEncoder
# from default_config import configs
from utils import get_ids, logit_normal_sample, euler_sampler, euler_sampler_test


class FluxLightning(L.LightningModule):
    def __init__(
        self,
        params:FluxParams,
        learning_rate,
        max_lr,
        warmup_pct,
        weight_decay,
        beta1,
        beta2
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = 1000
        
        # spec = configs["flux-schnell"]
        self.params = params
        self.flux = Flux(self.params)
        
        self.cond_encoder = ImageEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            out_channels=512
        )
        self.flux.vector_in = None
        
        # Initialize the autoencoder
        self.autoencoder = HuggingFaceAutoEncoder()
        
        pdf, x = logit_normal_sample(mu=1, sigma=1, num_samples=self.steps*10)
        self.register_buffer('pdf', pdf)
        self.register_buffer('x', x)
        # self.register_buffer('u_pdf', u_pdf)
        # self.register_buffer('u_x', u_x)

        self.print_hyperparameters()
        
    def print_hyperparameters(self):
        print("\n===== FluxLightning Hyperparameters =====")
        print(f"learning_rate: {self.learning_rate}")
        print(f"max_lr:        {self.max_lr}")
        print(f"warmup_pct:    {self.warmup_pct}")
        print(f"weight_decay:  {self.weight_decay}")
        print(f"beta1:         {self.beta1}")
        print(f"beta2:         {self.beta2}")
        print(f"steps:         {self.steps}")
        print("========================================\n")
        
    def sample_timesteps(self, batch_size, eps=1e-4):
        t = torch.distributions.Beta(concentration1=2.5, concentration0=2.0).sample((batch_size,)).to(self.device)
        t = t.clamp(eps, 1.0 - eps)                        # (0, 1) guard-band
        assert torch.all((t >= 0.0) & (t <= 1.0)), "t out of [0,1] range"
        return t
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2)
        )
        return optimizer
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.max_lr,
        #     total_steps=self.trainer.estimated_stepping_batches,
        #     pct_start=self.warmup_pct,
        # )
        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
    def forward(self, x: Tensor, x_ids: Tensor, y: Tensor, y_ids: Tensor, timesteps: Tensor) -> Tensor:
        # vec = torch.zeros(x.shape[0], self.params.hidden_size, device=self.device)
        return self.flux(img=x, img_ids=x_ids, txt=y, txt_ids=y_ids, timesteps=timesteps, y=None)
        
    def prepare(self, batch):
        cond, target = batch
        b, c, h, w = target.shape
        _b, _c, _h, _w = cond.shape
        assert _c == 3, f"Condition must have 3 channels, got {_c}"
        assert h == w == 256, f"Target image must be 256x256, got {h}x{w}"
        y = self.cond_encoder(cond)        
        y = y.view(b, 512, -1).permute(0, 2, 1).contiguous()
        with torch.no_grad():
            target_encoded = self.autoencoder.encode(target)
        z0 = target_encoded.view(b, 4, -1).permute(0, 2, 1).contiguous()

        x_ids, y_ids = get_ids(32, 32, y.shape[1])  # Use 32x32 for encoded dimensions
        x_ids = repeat(x_ids, 'a l c -> (a b) l c', b=b).to(self.device)
        y_ids = repeat(y_ids, 'a l c -> (a b) l c', b=b).to(self.device)
        
        timesteps = self.sample_timesteps(b).to(self.dtype)
        z1 = torch.randn_like(z0)
        ts = timesteps.reshape(b, 1, 1)
        zt = (1 - ts) * z0 + ts * z1
        
        return {
            'x': zt, 'y': y,
            'x_ids': x_ids, 'y_ids': y_ids,
            'timesteps': timesteps,
            'noise': z1, 'z0': z0,
        }
        
    def training_step(self, batch, batch_idx):
        data = self.prepare(batch)
        z1, z0 = data['noise'], data['z0']
        
        v_theta = self.forward(
            x=data['x'], x_ids=data['x_ids'],
            y=data['y'], y_ids=data['y_ids'],
            timesteps=data['timesteps']
        )
        
        v = z1 - z0
        mse = F.mse_loss(v_theta, v)
        
        self.log('train_loss', mse, sync_dist=True, prog_bar=True)
        return mse
        
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            data = self.prepare(batch)
            z1, z0 = data['noise'], data['z0']
            
            v_theta = self.forward(
                x=data['x'], x_ids=data['x_ids'],
                y=data['y'], y_ids=data['y_ids'],
                timesteps=data['timesteps']
            )
            
            v = z1 - z0
            loss = F.mse_loss(v_theta, v)
            self.log('val_loss', loss, sync_dist=True, prog_bar=True)
            
            ts_losses = (v_theta - v).pow(2).mean(dim=(1, 2))
            return {
                'timesteps': data['timesteps'].cpu().numpy(),
                'ts_losses': ts_losses.cpu().numpy(),
                'loss': loss,
            }

    @torch.no_grad()
    def sample(self, cond: Tensor, steps: int=24) -> Tensor:
        b, c, h, w = cond.shape
        assert h == w == 256, f"Condition must be 256x256, got {h}x{w}"
        assert c == 3, f"Condition must have 3 channels, got {c}"
        
        y = self.cond_encoder(cond.to(self.device))
        y = y.view(b, 512, -1).permute(0, 2, 1).contiguous()
        x_ids, y_ids = get_ids(32, 32, y.shape[1])
        x_ids = repeat(x_ids, 'a l c -> (a b) l c', b=b).to(self.device)
        y_ids = repeat(y_ids, 'a l c -> (a b) l c', b=b).to(self.device)

        z1 = torch.randn((b, 32*32, 4), device=self.device, dtype=self.dtype)
        zt = euler_sampler(self, y, z1, x_ids, y_ids, steps)

        zt = zt.permute(0, 2, 1).view(b, 4, 32, 32).contiguous()
        zt = self.autoencoder.decode(zt)

        decoded_img = zt.clamp(0, 1)
        return decoded_img

    @torch.no_grad()
    def sample_test(self, cond: Tensor, steps: int=24) -> Tensor:
        b, c, h, w = cond.shape
        assert h == w == 256, f"Condition must be 256x256, got {h}x{w}"
        assert c == 3, f"Condition must have 3 channels, got {c}"
        
        y = self.cond_encoder(cond.to(self.device))
        y = y.view(b, 512, -1).permute(0, 2, 1).contiguous()
        x_ids, y_ids = get_ids(32, 32, y.shape[1])
        x_ids = repeat(x_ids, 'a l c -> (a b) l c', b=b).to(self.device)
        y_ids = repeat(y_ids, 'a l c -> (a b) l c', b=b).to(self.device)

        z0 = self.autoencoder.encode(cond)
        z0 = z0.view(b, 4, -1).permute(0, 2, 1).contiguous()

        z1 = torch.randn((b, 32*32, 4), device=self.device, dtype=self.dtype)
        zt = euler_sampler_test(self, y, z1, z0, x_ids, y_ids, steps)

        print(f'Final Loss: {F.mse_loss(zt, z0)}')

        z0 = z0.permute(0, 2, 1).view(b, 4, 32, 32).contiguous()
        z0 = self.autoencoder.decode(z0)
        decoded_input = z0.clamp(0, 1).squeeze(0)

        zt = zt.permute(0, 2, 1).view(b, 4, 32, 32).contiguous()
        zt = self.autoencoder.decode(zt)

        decoded_img = zt.clamp(0, 1).squeeze(0)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(decoded_input.cpu().numpy().transpose(1, 2, 0))
        axs[1].imshow(decoded_img.cpu().numpy().transpose(1, 2, 0))
        plt.savefig('decoded_img.png', dpi=300)
        plt.close("all")
        return decoded_img

def test_flux_lightning():
    model = FluxLightning()
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    batch_size = 2
    
    cond = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.float32)
    target = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.float32)
    batch = (cond, target)
    
    loss = model.training_step(batch, 0)
    val_result = model.validation_step(batch, 0)
    
    print(f"Training loss: {loss}")
    print(f"Validation loss: {val_result['loss']}")
    return loss, val_result

if __name__ == "__main__":
    test_flux_lightning()
        
        
