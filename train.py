import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataset import ImageNetDataModule
from model import ResNet50LightningModule
from config import CONFIG


def main():
    # Data module
    data_module = ImageNetDataModule()

    # Model
    model = ResNet50LightningModule()

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["epochs"],
        gpus=CONFIG["gpus"],
        precision=CONFIG["precision"],  # Mixed precision training
        callbacks=[checkpoint_callback, lr_monitor],
        check_val_every_n_epoch=CONFIG["check_val_every_n_epoch"],  # Run validation every 5 epochs
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
