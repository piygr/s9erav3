import pytorch_lightning as pl
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from config import CONFIG

scale = 1.1
IMAGE_SIZE = 224
class AlbumentationsTransform:
    """
    Wrapper for applying Albumentations transforms to a PIL image.
    """
    def __init__(self, p: float = 0.5, eval: bool = False):

        if eval:
            augs = [
                A.SmallestMaxSize(max_size=IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ]
        else:
            augs = [
                A.SmallestMaxSize(max_size=int(IMAGE_SIZE*scale)),
                A.PadIfNeeded(
                    min_height=int(IMAGE_SIZE*scale),
                    min_width=int(IMAGE_SIZE*scale),
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
                A.Rotate(limit=10, interpolation=1, border_mode=4, p=p),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=p),
                A.HorizontalFlip(p=p),
                A.Blur(p=0.1),
                A.CLAHE(p=0.1),
                A.Posterize(p=0.1),
                A.ToGray(p=0.1),
                A.ChannelShuffle(p=0.05),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=112,
                    max_width=112,
                    min_holes=1,
                    min_height=112,
                    min_width=112,
                    fill_value=(0.485, 0.456, 0.406),
                    mask_fill_value=None,
                    p=p,
                ),
            ]

        augs += [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        self.transform = A.Compose(augs)

    def __call__(self, image):
        image = np.array(image)
        return self.transform(image=image)["image"]

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        #self.data_dir = CONFIG["data_dir"]
        self.train_data_dir = CONFIG["train_data_dir"]
        self.val_data_dir = CONFIG["val_data_dir"]
        self.batch_size = CONFIG["batch_size"]
        self.num_workers = CONFIG["num_workers"]
        self.augment_prob = CONFIG["augment_prob"]

    def setup(self, stage: str = None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = datasets.ImageFolder(
                root=f"{self.train_data_dir}",
                transform=AlbumentationsTransform(p=self.augment_prob)
            )
            self.val_dataset = datasets.ImageFolder(
                root=f"{self.val_data_dir}",
                transform=AlbumentationsTransform(eval=True)  # No augmentations for validation
            )
        if stage == "test":
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.val_data_dir}",
                transform=AlbumentationsTransform(eval=True)  # No augmentations for testing
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=CONFIG["pin_memory"],
        )