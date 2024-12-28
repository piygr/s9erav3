import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.optim.lr_scheduler import OneCycleLR
from config import CONFIG
from torch_lr_finder import LRFinder


class ResNet50LightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        # Load the pretrained ResNet50 model and modify the final layer
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, CONFIG["num_classes"])

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    '''def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            epochs=self.hparams.epochs,
            steps_per_epoch=len(self.train_dataloader()),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]'''

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=0.01)
        optimizer = torch.optim.SGD(self.parameters(),
                        lr=CONFIG["learning_rate"])
        self.max_lr = CONFIG["learning_rate"]
        self.find_lr(optimizer)
        print(self.max_lr)
        scheduler = OneCycleLR(optimizer,
                                max_lr=self.max_lr,
                                epochs=CONFIG["epochs"],
                                steps_per_epoch=len(self.train_dataloader()),
                                three_phase=True,
                                verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def find_lr(self, optimizer):
        if not CONFIG["lr_finder"]:
            return

        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(self.train_dataloader(), end_lr=10, num_iter=1000)
        _, best_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()
        self.max_lr = best_lr