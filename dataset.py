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


def handle_mixed_image(image, background_color=(255, 255, 255)):
    """
    Handle RGBA images by blending with a background color or converting to RGB.

    Args:
        image (np.ndarray): Input image as a NumPy array (from PIL).
        background_color (tuple): Background color to blend with, e.g., (255, 255, 255) for white.

    Returns:
        np.ndarray: RGB image as a NumPy array.
    """
    # Separate alpha channel and normalize it to [0, 1]
    alpha = image[:, :, 3] / 255.0  # Shape: (H, W)
    rgb = image[:, :, :3]  # Extract RGB channels

    # Create a background image with the specified color
    background = np.ones_like(rgb, dtype=np.float32) * np.array(background_color, dtype=np.float32)

    # Blend the image with the background based on alpha
    image = rgb * alpha[:, :, None] + background * (1 - alpha[:, :, None])

    # Convert the blended image to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image

def drop_alpha_if_exists(image_array):
    if len(image_array.shape) == 3 and image_array.shape[-1] == 4:  # RGBA
        return image_array[:, :, :3]  # Drop alpha
    return image_array


class AlbumentationsTransform:
    """
    Wrapper for applying Albumentations transforms to a PIL image.
    """
    def __init__(self, p: float = 0.5, eval: bool = False):

        if eval:
            augs = [
                A.LongestMaxSize(max_size=IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=IMAGE_SIZE,
                    min_width=IMAGE_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0.485 * 255, 0.456 * 255, 0.406 * 255)
                ),
            ]
        else:
            augs = [
                A.LongestMaxSize(max_size=int(IMAGE_SIZE*scale)),
                A.PadIfNeeded(
                    min_height=int(IMAGE_SIZE*scale),
                    min_width=int(IMAGE_SIZE*scale),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0.485 * 255, 0.456 * 255, 0.406 * 255)
                ),
                A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
                A.Rotate(limit=(-10,10), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=(0.485 * 255, 0.456 * 255, 0.406 * 255), p=p),
                A.HorizontalFlip(p=p),
                A.Blur(p=0.1),
                A.CoarseDropout(
                    num_holes_range=(1, 1),
                    hole_height_range = (112, 112),
                    hole_width_range = (112, 112),
                    fill=(0.485, 0.456, 0.406),
                    fill_mask=None,
                    p=p,
                ),
            ]

        augs += [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]

        self.transform = A.Compose(augs)



    def __call__(self, image):
        image = image.convert("RGB")
        image = np.array(image)
        '''if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = drop_alpha_if_exists(image)'''

        return self.transform(image=image)["image"]


class ImageNetDataset:
    def __init__(self, train=True, transform=None):
        self.transform = transform
        self.dataset = []
        self.train = train

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
                    first_value = line.split()[0].strip()
                    # Map the first value to the line number
                    self.class_mapping[first_value] = line_number

            if train:
                data_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/train"
                output_path = "./data_annotations_train.csv"
                with open(output_path, "a") as fw:
                    # Walk through the directory tree
                    for dirpath, dirnames, filenames in os.walk(data_dir):
                        for file in filenames:
                            # Get the relative path of the file
                            relative_path = os.path.relpath(os.path.join(dirpath, file), data_dir).strip()
                            image_id = file.split(".")[0].strip()
                            class_id = relative_path.split("/")[0].strip()
                            if class_id in self.class_mapping:
                                self.dataset.append((relative_path, self.class_mapping[class_id]))

                                # write to file
                                fw.write(f"{relative_path},{self.class_mapping[class_id]}\n")
            else:
                data_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/val"
                label_file = CONFIG["root_dir"] + "/LOC_val_solution.csv"

                df = pd.read_csv(label_file)

                # Convert mapping to a dictionary for better usability
                imageid_label_mapping_dict = {row[0].lower().strip(): self.class_mapping.get(row[1].split()[0].strip()) for _, row in df.iterrows()}

                output_path = "./data_annotations_val.csv"
                with open(output_path, "a") as fw:
                    # Walk through the directory tree
                    for dirpath, dirnames, filenames in os.walk(data_dir):
                        for file in filenames:
                            # Get the relative path of the file
                            relative_path = os.path.relpath(os.path.join(dirpath, file), data_dir).strip()
                            image_id = file.split(".")[0].strip()
                            if image_id.lower() in imageid_label_mapping_dict:
                                self.dataset.append((relative_path, imageid_label_mapping_dict[image_id.lower()]))

                                #write to file
                                fw.write(f"{relative_path},{imageid_label_mapping_dict[image_id.lower()]}\n")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_dir = CONFIG["root_dir"] + "/ILSVRC/Data/CLS-LOC/%s/" % ("train" if self.train else "val")
        image, label = self.dataset[idx]
        image = Image.open(image_dir + image)
        #image = np.array(image)  # Convert PIL Image to NumPy array
        if self.transform:
            image = self.transform(image=image)  # Albumentations transform
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