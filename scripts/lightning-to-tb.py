import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Define paths
csv_file = "./lightning_logs/version_0/metrics.csv"  # Replace with the path to your metrics.csv file
log_dir = "./tensorboard_logs"   # Directory to store TensorBoard logs

# Read the metrics.csv file
metrics = pd.read_csv(csv_file)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir)

# Log metrics to TensorBoard
for index, row in metrics.iterrows():
    # Log training metrics
    if not pd.isna(row.get("train_loss_step")):
        writer.add_scalar("Train/Loss", row["train_loss_step"], int(row["step"]))
    if not pd.isna(row.get("train_acc_step")):
        writer.add_scalar("Train/Accuracy", row["train_acc_step"], int(row["step"]))

    # Log validation metrics
    if not pd.isna(row.get("val_loss")):
        writer.add_scalar("Validation/Loss", row["val_loss"], int(row["step"]))
    if not pd.isna(row.get("val_acc")):
        writer.add_scalar("Validation/Accuracy", row["val_acc"], int(row["step"]))

# Close the writer
writer.close()

print(f"TensorBoard logs written to {log_dir}")

