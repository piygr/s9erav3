import os
import pytorch_lightning as pl
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

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
                    value=(0.485 * 255, 0.456 * 255, 0.406 * 255)
                ),
            ]
        else:
            augs = [
                A.SmallestMaxSize(max_size=int(IMAGE_SIZE*scale)),
                A.PadIfNeeded(
                    min_height=int(IMAGE_SIZE*scale),
                    min_width=int(IMAGE_SIZE*scale),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0.485 * 255, 0.456 * 255, 0.406 * 255)
                ),
                A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
                A.Rotate(limit=10, interpolation=1, border_mode=4, value=(0.485 * 255, 0.456 * 255, 0.406 * 255), p=p),
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


class ImageNetDataset:
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.dataset = []

        annot_file = True
        if CONFIG.get("data_annotation_file", {}):
            if train and CONFIG["data_annotation_file"]["train"]:
                df = pd.read_csv(CONFIG["data_annotation_file"]["train"])
                self.dataset = [(row[0], row[1]) for _, row in df.iterrows()]
            elif not train and CONFIG["data_annotation_file"]["val"]:
                df = pd.read_csv(CONFIG["data_annotation_file"]["val"])
                self.dataset = [ (row[0], row[1]) for _, row in df.iterrows() ]
            else:
                annot_file = False
        else:
            annot_file = False

        if not annot_file:

            class_mapping_path_file_path = CONFIG["root_dir"] + "/LOC_synset_mapping.txt"

            # Initialize an empty dictionary to store the mapping
            self.class_mapping = {}

            # Open the file and parse it
            with open(class_mapping_path_file_path, "r") as file:
                for line_number, line in enumerate(file):
                    # Split the line to get the first value
                    first_value = line.split()[0]
                    # Map the first value to the line number
                    self.class_mapping[first_value] = line_number

            data_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/train"
            label_file = CONFIG["root_dir"] + "/LOC_train_solution.csv"
            if not train:
                data_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/val"
                label_file = CONFIG["root_dir"] + "/LOC_val_solution.csv"

            df = pd.read_csv(label_file)

            # Convert mapping to a dictionary for better usability
            imageid_label_mapping_dict = {row[0].lower(): self.class_mapping.get(row[1].split()[0]) for _, row in df.iterrows()}

            output_path = "./data_annotations_%s.csv" % ("train" if train else "val")
            with open(output_path, "a") as fw:
                # Walk through the directory tree
                for dirpath, dirnames, filenames in os.walk(data_dir):
                    for file in filenames:
                        # Get the relative path of the file
                        relative_path = os.path.relpath(os.path.join(dirpath, file), data_dir)
                        image_id = file.split(".")[0]
                        if imageid_label_mapping_dict.get(image_id.lower()):
                            self.dataset.append((relative_path, imageid_label_mapping_dict[image_id.lower()]))

                            #write to file
                            fw.write(f"{relative_path},{imageid_label_mapping_dict[image_id.lower()]}\n")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = Image.open(image)
        image = np.array(image)  # Convert PIL Image to NumPy array
        if self.transform:
            augmented = self.transform(image=image)  # Albumentations transform
            image = augmented["image"]  # Extract the transformed image
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = CONFIG["batch_size"]
        self.num_workers = CONFIG["num_workers"]
        self.augment_prob = CONFIG["augment_prob"]

    def setup(self, stage: str = None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = ImageNetDataset(
                train=True,
                transform=AlbumentationsTransform(p=self.augment_prob)
            )
            self.val_dataset = ImageNetDataset(
                train=False,
                transform=AlbumentationsTransform(eval=True)  # No augmentations for validation
            )
        if stage == "test":
            self.test_dataset = ImageNetDataset(
                train=False,
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