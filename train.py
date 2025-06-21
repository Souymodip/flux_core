import lightning as L
from fluxlt import FluxLightning
from callbacks import get_callbacks
from lightning.pytorch.loggers import WandbLogger
import os
from dataloader import  create_polyline_dataloaders

def train(
    train_dataloader,
    val_dataloader,
    max_epochs=100,
    learning_rate=1e-3,
    max_lr=1e-3,
    warmup_pct=0.3,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
    save_dir='./checkpoints',
    grad_freq=256,
    n_samples=4
):
    model = FluxLightning(
        learning_rate=learning_rate,
        max_lr=max_lr,
        warmup_pct=warmup_pct,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2
    )
    
    callbacks = get_callbacks(grad_freq=grad_freq, n_samples=n_samples)
    logger = WandbLogger(
        project="flux-vae-test",
        save_dir=save_dir
    )
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        logger=logger
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    return model, trainer


if __name__ == "__main__":
    train_dataloader, val_dataloader = create_polyline_dataloaders(
        folder_path="/Users/souymodip/GIT/DATA_SETS/POLY1536_charac100",
        size=256,
        batch_size=2,
        num_workers=4,
        split=0.9
    )
    
    model, trainer = train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_epochs=100
    ) 