import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.tuner.tuning import Tuner
import torch
from dataset import ImageNetDataModule
from model import ResNet50LightningModule
from config import CONFIG

'''def find_optimal_lr(trainer, model, data_module):
    """
    Use PyTorch Lightning's Tuner to find the optimal learning rate.
    """
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=data_module)
    optimal_lr = lr_finder.suggestion()
    print(f"Suggested Learning Rate: {optimal_lr}")
    lr_finder.plot(suggest=True).show()
    return optimal_lr'''


def main():
    # Data module
    data_module = ImageNetDataModule()

    # Model
    model = ResNet50LightningModule()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    '''
    # Trainer for LR Finder
    lr_finder_trainer = pl.Trainer(
        max_epochs=1,
        gpus=-1 if torch.cuda.is_available() else None,  # Use all GPUs if available
        auto_lr_find=True,
    )

    # Find the optimal learning rate
    optimal_lr = find_optimal_lr(lr_finder_trainer, model, data_module)

    # Update the model with the optimal learning rate
    model.hparams.learning_rate = optimal_lr'''

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu",
        devices="auto",
        precision=CONFIG["precision"],  # Mixed precision training
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=CONFIG["check_val_every_n_epoch"],  # Run validation every 5 epochs
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
