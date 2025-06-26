import lightning as L
from fluxlt import FluxLightning
from callbacks import get_callbacks
from lightning.pytorch.loggers import WandbLogger
from dataloader import  create_polyline_dataloaders
import torch
from model import FluxParams
import numpy as np

torch.set_float32_matmul_precision('medium')

params=FluxParams(
    in_channels=4,
    out_channels=4,
    vec_in_dim=1,
    context_in_dim=512,
    hidden_size=1024,
    mlp_ratio=4.0,
    num_heads=8,
    depth=24,
    depth_single_blocks=12,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=False,
)

def train(
    train_dataloader,
    val_dataloader,
    max_epochs=100,
    learning_rate=1e-4,
    max_lr=1e-3,
    warmup_pct=0.3,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
    save_dir='/mnt/localssd/SAVES/FLUX_CORE/flux-i2i-test',
    grad_freq=512,
    n_samples=4
):
    model = FluxLightning(
        params=params,
        learning_rate=learning_rate,
        max_lr=max_lr,
        warmup_pct=warmup_pct,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2
    )
    
    callbacks = get_callbacks(dirpath=save_dir, grad_freq=grad_freq, n_samples=n_samples)
    logger = WandbLogger(
        project="f-s1024",
        save_dir=save_dir
    )
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        # devices=1, #'auto',
        # precision='bf16-mixed',
        # limit_val_batches=64, #val_check_interval=0.2,
        accumulate_grad_batches=8,
        logger=logger,
        # limit_train_batches=8,
    )
    ckpt_path = None #"/mnt/localssd/SAVES/FLUX_CORE/flux-i2i-test/last.ckpt"
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    return model, trainer

def test_sample_from_ckpt(ckpt_path:str, img:torch.Tensor):
    model = FluxLightning.load_from_checkpoint(ckpt_path, params=params, learning_rate=1e-4, max_lr=1e-3, warmup_pct=0.3, weight_decay=0.01, beta1=0.9, beta2=0.999)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    img = img.to(device)
        
    with torch.no_grad():
        out = model.sample_test(img, steps=24)
    return out.cpu().numpy()

if __name__ == "__main__":
    train_dataloader, val_dataloader = create_polyline_dataloaders(
        folder_path="/mnt/localssd/Poly1536/",
        size=256,
        batch_size=4,
        num_workers=8,
        split=0.9999
    )
    
    # model, trainer = train(
    #     train_dataloader=train_dataloader,
    #     val_dataloader=train_dataloader,
    #     max_epochs=100
    # ) 

    cond, target = next(iter(train_dataloader))
    out = test_sample_from_ckpt(ckpt_path="/mnt/localssd/SAVES/FLUX_CORE/flux-i2i-test/epoch=61val_loss=0.0058.ckpt", img=cond[[0]])