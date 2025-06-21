import lightning as L
import torch
import torchvision
import os
from fluxlt import FluxLightning
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import io
from PIL import Image

class ImageLogger(L.Callback):
    def __init__(self, n_samples: int=4):
        super().__init__()
        self.n_samples = n_samples
        self.timestep_data = np.zeros((24, 2))

    @torch.no_grad()
    def on_validation_end(self, trainer: L.Trainer, pl_module: FluxLightning) -> None:
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
        
        decoded_img = pl_module.sample(cond, steps=24).cpu()
        
        # Create grid with condition on top row and decoded image on bottom row
        # Stack condition and decoded images vertically
        combined_images = torch.cat([cond, decoded_img], dim=0)
        
        # Create grid
        grid = torchvision.utils.make_grid(
            combined_images, 
            nrow=self.n_samples,  # Number of images per row
            padding=4
        )
        
        try:
            # Log the grid image
            trainer.logger.experiment.log({ 'condition_vs_decoded': grid }, commit=True)
        except Exception as e:
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.savefig('grid.png')
            plt.close("all")
            import pdb; pdb.set_trace()
            print(f"Error logging image: {e}")

        if np.sum(self.timestep_data[:, 1]) > 0:
            buf = io.BytesIO()
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), tight_layout=True)
            avg_losses = np.zeros(self.timestep_data.shape[0])
            counts = self.timestep_data[:, 1]
            avg_losses[counts > 0] = self.timestep_data[counts > 0, 0] / counts[counts > 0  ]
            axs[0].bar(np.arange(self.timestep_data.shape[0]), avg_losses)
            axs[0].set_title('Average Loss per Timestep')
            axs[1].bar(np.arange(self.timestep_data.shape[0]), counts)
            axs[1].set_title('Count of Samples per Timestep')
            plt.savefig(buf)
            image = np.array(Image.open(buf))
            buf.seek(0)
            plt.close(fig)
            try:
                trainer.logger.experiment.log({ 'timestep_avg_loss_vs_count': image }, commit=True)
            except Exception as e:
                plt.imshow(image)
                plt.savefig('timestep_avg_loss_vs_count.png')
                plt.close("all")
                import pdb; pdb.set_trace()
                print(f"Error logging histogram: {e}")
            
            self.timestep_data = np.zeros((24, 2))

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
        print(f"global_step: {trainer.global_step}, grad_freq: {self.grad_freq}, global_rank: {trainer.global_rank}")
        if trainer.global_step % self.grad_freq != 0 or trainer.global_rank != 0:
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
                        


class ModelCheckpoint(L.Callback):
    def __init__(self, dirpath="checkpoints", filename="flux-{epoch:02d}-{val_loss:.4f}", save_top_k=3, monitor="val/batch_loss", mode="min"):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = None
        self.best_model_path = ""
        
        os.makedirs(dirpath, exist_ok=True)
        
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        
        if self.monitor not in logs:
            return
            
        current = logs[self.monitor].item()
        
        if self.best_model_score is None:
            self.best_model_score = current
            self.best_model_path = os.path.join(self.dirpath, self.filename.format(epoch=epoch, val_loss=current))
            trainer.save_checkpoint(self.best_model_path)
            return
            
        if (self.mode == "min" and current < self.best_model_score) or (self.mode == "max" and current > self.best_model_score):
            self.best_model_score = current
            self.best_model_path = os.path.join(self.dirpath, self.filename.format(epoch=epoch, val_loss=current))
            trainer.save_checkpoint(self.best_model_path)


class LearningRateMonitor(L.Callback):
    def __init__(self, logging_interval="step"):
        super().__init__()
        self.logging_interval = logging_interval
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.logging_interval == "step":
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            trainer.logger.log_metrics({"lr": current_lr})


def get_callbacks(grad_freq: int=2, n_samples: int=4):
    return [
        ImageLogger(n_samples),
        GradientLogger(grad_freq),
        ModelCheckpoint(),
        LearningRateMonitor()
    ]