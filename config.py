CONFIG = {
    # General Configuration
    "data_dir": "/path/to/imagenet",  # Path to ImageNet dataset
    "num_classes": 1000,  # Number of prediction classes
    "batch_size": 128,  # Batch size for training
    "epochs": 40,  # Total number of epochs
    "learning_rate": 1e-3,  # Initial learning rate
    "augment_prob": 0.5,  # Probability for augmentations (e.g., HorizontalFlip)

    # Hardware Configuration
    "gpus": 1,  # Number of GPUs to use
    "precision": 16,  # FP16 mixed precision training

    # DataLoader Configuration
    "num_workers": 4,  # Number of DataLoader workers
    "pin_memory": True,  # Use pinned memory for DataLoader

    "check_val_every_n_epoch": 5
}