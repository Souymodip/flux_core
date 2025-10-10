import lightning as L
import torch
import torchvision
import os
from fluxlt import FluxLightning, get_ids
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from einops import repeat

class ImageLogger(L.Callback):
    def __init__(self, n_samples: int=4):
        super().__init__()
        self.n_samples = n_samples
        self.timestep_data = np.zeros((24, 2))

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: FluxLightning) -> None:
        if trainer.global_rank != 0 or trainer.current_epoch % 10 != 0:
            return
        
        val_dataloader = trainer.val_dataloaders
        dataset_size = len(val_dataloader.dataset)
        assert dataset_size >= self.n_samples, f"Dataset size {dataset_size} must be greater than or equal to n_samples {self.n_samples}"
        indices = torch.randperm(dataset_size)[:self.n_samples]
        
        # Collect condition tensors from multiple samples
        cond_list = []
        for idx in indices:
            cond, _ = val_dataloader.dataset[idx]  # Extract condition from tuple
            cond_list.append(cond)
        # Concatenate to create batch
        cond = torch.stack(cond_list, dim=0)
        # Move cond to the same device as pl_module
        cond = cond.to(pl_module.device)
        # Use VAE to encode and decode cond
        cond_encoded = pl_module.autoencoder.encode(cond)
        cond_decoded = pl_module.autoencoder.decode(cond_encoded).cpu()
        
        decoded_img = pl_module.sample(cond, steps=24).cpu()
        
        # Create grid with VAE-decoded condition on top row and decoded image on bottom row
        combined_images = torch.cat([cond_decoded, decoded_img], dim=0)
        
        # Create grid
        grid = torchvision.utils.make_grid(
            combined_images, 
            nrow=self.n_samples,  # Number of images per row
            padding=4
        )
        
        try:
            # Log the grid image
            trainer.logger.experiment.log({ 'condition_vs_decoded':  wandb.Image(grid) }, commit=True)
        except Exception as e:
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig('grid.png')
            plt.close("all")
            import pdb; pdb.set_trace()
            print(f"Error logging image: {e}")

        if np.sum(self.timestep_data[:, 1]) > 0:
            buf = io.BytesIO()
            fig, axs = plt.subplots(2, 1, figsize=(8, 4), tight_layout=True)
            avg_losses = np.zeros(self.timestep_data.shape[0])
            counts = self.timestep_data[:, 1]
            avg_losses[counts > 0] = self.timestep_data[counts > 0, 0] / counts[counts > 0  ]
            axs[0].bar(np.arange(self.timestep_data.shape[0]), avg_losses)
            axs[0].set_title('Average Loss per Timestep')
            axs[1].bar(np.arange(self.timestep_data.shape[0]), counts)
            axs[1].set_title('Count of Samples per Timestep')
            fig.savefig(buf)
            buf.seek(0)
            plt.close(fig)
            self.timestep_data = np.zeros((24, 2))
            image = np.array(Image.open(buf), dtype=np.uint8)
            try:
                trainer.logger.experiment.log({ 'timestep_avg_loss_vs_count':  wandb.Image(image) }, commit=True)
            except Exception as e:
                plt.imshow(image)
                plt.savefig('timestep_avg_loss_vs_count.png')
                plt.close("all")
                import pdb; pdb.set_trace()
                print(f"Error logging histogram: {e}")
            
    @torch.no_grad()    
    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: FluxLightning, outputs: Dict[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if outputs is None:
            return
            
        timesteps = outputs.get('timesteps', None)
        ts_losses = outputs.get('ts_losses', None)
        bins = self.timestep_data.shape[0]
        if timesteps is not None and ts_losses is not None:
            for ts, loss in zip(timesteps, ts_losses):
                ts_bin = int(ts * bins)
                ts_bin = min(ts_bin, bins - 1)
                self.timestep_data[ts_bin, 0] += loss
                self.timestep_data[ts_bin, 1] += 1


class GradientLogger(L.Callback):
    def __init__(self, grad_freq: int=100):
        super().__init__()
        self.grad_freq = grad_freq

        self.module_names = set(['pe_embedder', 'img_in', 'time_in', 'txt_in', 
                        'double_blocks', 'single_blocks', 'final_layer'])
        
    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.global_step ==0 or trainer.global_step % self.grad_freq != 0 or trainer.global_rank != 0:
            return
        
        for name, flux_module in pl_module.named_modules():
            if name == 'flux': 
                for name, module in flux_module.named_modules():
                    if name not in self.module_names: continue
                    sum_grad_norms, grad_norm_count = 0., 0
                    for param in module.parameters():
                        if param.requires_grad and param.grad is not None:
                            sum_grad_norms += param.grad.norm().item()
                            grad_norm_count += 1
                    if grad_norm_count > 0:
                        pl_module.log(f'grad_norm/{name}', sum_grad_norms / grad_norm_count, sync_dist=True)
            if name == 'cond_encoder':
                sum_grad_norms, grad_norm_count = 0., 0
                for param in flux_module.parameters():
                    if param.requires_grad and param.grad is not None:
                        sum_grad_norms += param.grad.norm().item()
                        grad_norm_count += 1
                if grad_norm_count > 0:
                    pl_module.log(f'grad_norm/cond_encoder', sum_grad_norms / grad_norm_count, sync_dist=True)
                        


class LearningRateMonitor(L.Callback):
    def __init__(self, logging_interval="step"):
        super().__init__()
        self.logging_interval = logging_interval
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.logging_interval == "step":
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            trainer.logger.log_metrics({"lr": current_lr})


class TrainImageLogger(L.Callback):
    def __init__(self, n_samples: int=4):
        super().__init__()
        self.n_samples = n_samples

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: FluxLightning) -> None:
        if trainer.global_rank != 0 or trainer.current_epoch % 10 != 0:
            return
        train_dataloader = getattr(trainer, 'train_dataloader', None)
        if callable(train_dataloader):
            train_dataloader = train_dataloader()
        if train_dataloader is None:
            return
        if isinstance(train_dataloader, (list, tuple)):
            train_dataloader = train_dataloader[0]
        dataset = getattr(train_dataloader, 'dataset', None)
        if dataset is None:
            return
        dataset_size = len(dataset)
        assert dataset_size >= self.n_samples, f"Dataset size {dataset_size} must be greater than or equal to n_samples {self.n_samples}"
        indices = torch.randperm(dataset_size)[:self.n_samples]
        cond_list = []
        for idx in indices:
            cond, _ = dataset[idx]
            cond_list.append(cond)
        cond = torch.stack(cond_list, dim=0)
        # Move cond to the same device as pl_module
        cond = cond.to(pl_module.device)
        # Use VAE to encode and decode cond
        cond_encoded = pl_module.autoencoder.encode(cond)
        cond_decoded = pl_module.autoencoder.decode(cond_encoded).cpu()
        decoded_img = pl_module.sample(cond, steps=24).cpu()
        # Create grid with VAE-decoded condition on top row and decoded image on bottom row
        combined_images = torch.cat([cond_decoded, decoded_img], dim=0)
        grid = torchvision.utils.make_grid(combined_images, nrow=self.n_samples, padding=4)
        try:
            trainer.logger.experiment.log({ 'train_condition_vs_decoded':  wandb.Image(grid) }, commit=True)
        except Exception as e:
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig('train_grid.png')
            plt.close("all")
            import pdb; pdb.set_trace()
            print(f"Error logging train image: {e}")
    
    @torch.no_grad()
    def on_train_batch_end_mse_plot(self, trainer: L.Trainer, pl_module: FluxLightning, batch: Any) -> None:
        # Only run on rank 0 and first batch
        if trainer.global_rank != 0 or trainer.current_epoch % 10 != 0:
            return
        cond, target = batch
        if isinstance(cond, torch.Tensor) and cond.dim() == 4:
            cond = cond[:1]  # Take first sample
        if isinstance(target, torch.Tensor) and target.dim() == 4:
            target = target[:1]
        b, c, h, w = target.shape
        _b, _c, _h, _w = cond.shape
        assert _c == 3, f"Condition must have 3 channels, got {_c}"
        assert h == w == 256, f"Target image must be 256x256, got {h}x{w}"

        y = pl_module.cond_encoder(cond.to(pl_module.device))
        y = y.view(b, 512, -1).permute(0, 2, 1).contiguous()
        with torch.no_grad():
            target_encoded = pl_module.autoencoder.encode(target.to(pl_module.device))
        z0 = target_encoded.view(b, 4, -1).permute(0, 2, 1).contiguous()


        x_ids, y_ids = get_ids(32, 32, y.shape[1])
        x_ids = repeat(x_ids, 'a l c -> (a b) l c', b=b).to(pl_module.device)
        y_ids = repeat(y_ids, 'a l c -> (a b) l c', b=b).to(pl_module.device)

        ts_arr = np.linspace(1, 0, 24)
        mse_losses = []
        # Ensure dtype is a valid torch dtype
        dtype = getattr(pl_module, 'dtype', torch.float32)
        for ts in ts_arr:
            ts_tensor = torch.full((b, 1, 1), float(ts), device=pl_module.device, dtype=dtype)
            z1 = torch.randn_like(z0)
            zt = (1 - ts_tensor) * z0 + ts_tensor * z1
            timesteps_tensor = torch.full((b,), float(ts), device=pl_module.device, dtype=dtype)
            v_theta = pl_module.forward(
                x=zt, x_ids=x_ids, y=y, y_ids=y_ids, timesteps=timesteps_tensor
            )
            v = z1 - z0
            mse = torch.nn.functional.mse_loss(v_theta, v).item()
            mse_losses.append(mse)
        # Plot
        plt.figure(figsize=(6, 4))
        plt.plot(ts_arr, mse_losses, marker='o')
        plt.xlabel('Timestep (ts)')
        plt.ylabel('MSE Loss')
        plt.title('MSE Loss vs Timestep (Train Sample)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close()
        image = np.array(Image.open(buf), dtype=np.uint8)
        try:
            trainer.logger.experiment.log({'train_mse_vs_timestep': wandb.Image(image)}, commit=True)
        except Exception as e:
            plt.imshow(image)
            plt.savefig('train_mse_vs_timestep.png')
            plt.close("all")
            print(f"Error logging train mse plot: {e}")

    @torch.no_grad()
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: FluxLightning, outputs: Dict[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if trainer.global_rank != 0 or batch_idx != 0 or trainer.current_epoch % 10 != 0:
            return
        self.on_train_batch_end_mse_plot(trainer, pl_module, batch)

def get_callbacks(dirpath:str, grad_freq:int, n_samples:int):
    return [
        ImageLogger(n_samples),
        # GradientLogger(grad_freq),
        ModelCheckpoint(
            dirpath=dirpath,
            filename="{epoch:02d}{val_loss:.4f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(),
        TrainImageLogger(n_samples)
    ]