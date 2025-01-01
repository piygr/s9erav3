# ResNet50 Training on ImageNet-1k Dataset

This repository contains the implementation of training a **ResNet50** model on the **ImageNet 1k dataset** using **PyTorch**. The project utilizes the **OneCycleLR Policy** for dynamic learning rate scheduling, ensuring faster convergence and efficient training. The training is performed over **40 epochs** with a **batch size of 384**, and the maximum learning rate is set to **0.8** (calculated using `LRFinder` and extrapolated for batch size 384).

---

## **Features**
- **Dataset**: ImageNet 1k (1,281,167 training images across 1,000 classes).
- **Model Architecture**: ResNet50.
- **Learning Rate Scheduler**: OneCycleLR policy.
- **Optimization**:
  - **Optimizer**: SGD with momentum.
  - **Loss Function**: CrossEntropyLoss.
- **Batch Size**: 384 (extrapolated from learning rate calculation for batch size 256).
- **Precision**: Mixed precision training using `torch.cuda.amp` for improved efficiency.
- **Validation**: Evaluates the model after every 5th epoch to monitor progress.
- **Framework**: PyTorch and PyTorch Lightning.

---

## **Requirements**

### **Hardware**
- GPU with at least **24GB VRAM** (e.g., NVIDIA A100 or V100).
- CPU with sufficient threads for data loading and preprocessing.

### **Software**
- Python 3.8 or later.
- PyTorch and PyTorch Lightning.
- torchvision for datasets and transformations.

### **Dependencies**
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Clone the Repository**
```bash
git clone https://github.com/piygr/s9erav3.git
cd s9erav3
```

### **2. Dataset Setup**
Download and prepare the ImageNet 1k dataset from following:
1. https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
2. Update the `data_dir` path in the training script to point to your dataset directory.

### **3. Training**
Run the training script:
```bash
python train.py
```

### **4. Demo **
Here is the Space app link of the trained model -
- https://huggingface.co/spaces/piyushgrover/ResNet50
---

## **Training Configuration**

### **Learning Rate Finder**
- Used PyTorch Lightning's `lr_find` method to determine the learning rate for a batch size of **256**.
- Extrapolated the learning rate for batch size **384** using the formula:
 ![IMG_9038](https://github.com/user-attachments/assets/72097da5-622e-4efa-b908-ebfb4109aa98)

  - Base LR: `0.533` (from LRFinder).
  - Batch Size Base: `256`.
  - Batch Size New: `384`.
  - Extrapolated Max LR: `0.8`.

### **OneCycleLR Policy**
- **Scheduler**: OneCycleLR.
- **Max LR**: `0.8`.
- **Total Epochs**: `40`.
- **Warm-Up**: First 30% of steps.
- **Annealing Strategy**: Cosine.

<img width="797" alt="Screenshot 2025-01-01 at 2 02 07 PM" src="https://github.com/user-attachments/assets/3c550a27-fb75-46a4-ae22-a9902b045301" />

### **Batch Size**
- Batch Size: `384` (optimized for 24GB VRAM).

---

## **Results**
- **Validation Accuracy**: 73% Top-1 accuracy achieved during training.
- **Training Loss**: Demonstrates smooth convergence due to OneCycleLR.

<img width="625" alt="Screenshot 2025-01-01 at 2 02 31 PM" src="https://github.com/user-attachments/assets/61cabe6f-1954-4cb5-ba89-4d2a745c8f47" />

---

## **References**
- [ImageNet](http://www.image-net.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [OneCycleLR Policy](https://arxiv.org/abs/1708.07120)

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Contact**
For issues, questions, or suggestions, please open an issue in the repository or contact the author.

