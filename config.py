CONFIG = {
    # General Configuration
    "train_data_dir": "/kaggle/input/imagenet1k-subset-100k-train-and-10k-val/imagenet_subtrain",  # Path to ImageNet dataset
    "val_data_dir": "/kaggle/input/imagenet1k-subset-100k-train-and-10k-val/imagenet_subval",
    "num_classes": 1000,  # Number of prediction classes
    "batch_size": 128,  # Batch size for training
    "epochs": 40,  # Total number of epochs
    "learning_rate": 1e-3,  # Initial learning rate
    "momentum": 0.9,
    "weight_decay": 0.01,
    "augment_prob": 0.5,  # Probability for augmentations (e.g., HorizontalFlip)

    # Hardware Configuration
    "gpus": 1,  # Number of GPUs to use
    "precision": 16,  # FP16 mixed precision training

    # DataLoader Configuration
    "num_workers": 4,  # Number of DataLoader workers
    "pin_memory": True,  # Use pinned memory for DataLoader

    "check_val_every_n_epoch": 5,

    "lr_finder": True
}
