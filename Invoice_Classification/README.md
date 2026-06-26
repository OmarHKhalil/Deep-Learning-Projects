**Project: Image Intelligence Filter**

This repository contains the Image Intelligence Filter developed to act as a high-precision gateway that routes only relevant images (receipts) to downstream processing (e.g., bounding box detection). The filter classifies input images into three categories: `Invoice`, `Menu`, and `Irrelevant`.

**Workflow diagram**: [Flowchart: Workflow for image type validation using the ResNet-50 classifier](Images/Workflow%20for%20image%20type%20validation%20using%20the%20ResNet-50%20classifier..png)

1. Dataset

- Size & sources: Hybrid collection of 4,200 images (2.5 GB) assembled from Kaggle, ReceiptSense, and other public sources.
- Classes: Balanced across `Invoice`, `Menu`, and `Irrelevant`.
- Preprocessing & augmentations used:
  - Resize to 448×448 pixels.
  - Convert to grayscale while keeping 3 channels (to preserve lighting characteristics).
  - ColorJitter for brightness/contrast variation.
  - RandomRotation up to ±15 degrees.
  - Convert to tensors and normalize with standard mean/std values.

2. Models and Training

2.1 ResNet-50

- We experimented with transfer learning strategies: baseline (no fine-tuning), frozen-backbone (rebuild classification head), and full fine-tuning (unfreeze backbone after initial head training).
- Typical training schedule for the full fine-tuning strategy:
  - Freeze backbone for the first 5 epochs to adapt the new classification head.
  - Unfreeze all layers and fine-tune for epochs 6–25 with a lower learning rate.
- The frozen strategy was also evaluated over 35 epochs before early stopping.

**ResNet-50 training progress (Full fine-tuning strategy)**

| Phase       | Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----------- | ----- | ---------- | --------- | -------- | ------- |
| Frozen      | 1     | 0.4937     | 84.43%    | 0.1897   | 95.14%  |
| Frozen      | 3     | 0.2588     | 91.90%    | 0.1032   | 97.79%  |
| Frozen      | 5     | 0.1973     | 93.63%    | 0.0809   | 98.53%  |
| Fine-tuning | 6     | 0.1485     | 94.88%    | 0.0545   | 98.53%  |
| Fine-tuning | 10    | 0.0679     | 97.64%    | 0.0261   | 99.41%  |
| Fine-tuning | 16    | 0.0270     | 99.26%    | 0.0125   | 99.56%  |
| Fine-tuning | 21    | 0.0145     | 99.74%    | 0.0088   | 99.56%  |
| Fine-tuning | 25    | 0.0115     | 99.74%    | 0.0078   | 99.71%  |

**ResNet-50 training progress (Frozen strategy)**

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
| ----- | ---------- | -------------- | -------- | ------------ |
| 1     | 0.5134     | 85.79%         | 0.2544   | 95.14%       |
| 5     | 0.1405     | 96.50%         | 0.1117   | 96.91%       |
| 10    | 0.1000     | 96.91%         | 0.0697   | 97.79%       |
| 15    | 0.0747     | 97.86%         | 0.0486   | 98.82%       |
| 20    | 0.0635     | 98.23%         | 0.0605   | 98.23%       |
| 25    | 0.0528     | 98.34%         | 0.0439   | 98.53%       |
| 28    | 0.0535     | 98.45%         | 0.0391   | 99.12%       |
| 30    | 0.0508     | 98.75%         | 0.0379   | 98.82%       |
| 35    | 0.0518     | 98.31%         | 0.0434   | 98.23%       |

Key ResNet-50 assets:

- Confusion matrix: [Images/Resnet50_Confusion%20Matrix.png](Images/Resnet50_Confusion%20Matrix.png)
- Workflow diagram (above)

  2.2 YOLOv11-cls

- Trained from `yolo11l-cls.pt` pretrained weights with input size 448×448 and batch size 4.
- RandAugment was used to improve generalization; training ran for 20 epochs with weight_decay=0.0005 and the automatic optimizer selection.

**YOLOv11-cls training progress**

| Epoch | Train Loss | Val Loss | Top-1 Accuracy |
| ----- | ---------- | -------- | -------------- |
| 1     | 0.30111    | 0.09695  | 0.9750         |
| 5     | 0.18614    | 0.33788  | 0.9025         |
| 10    | 0.11408    | 0.09923  | 0.9475         |
| 15    | 0.06333    | 0.23375  | 0.9525         |
| 20    | 0.03373    | 0.11620  | 0.9850         |

Key YOLOv11 assets:

- Training & validation plot: [Images/Training%20and%20validation%20results%20for%20the%20YOLOv11-cls%20model..png](Images/Training%20and%20validation%20results%20for%20the%20YOLOv11-cls%20model..png)
- Confusion matrix and run outputs: [Images/runs_YOLO11/classify/train7/confusion_matrix.png](Images/runs_YOLO11/classify/train7/confusion_matrix.png)
- Example validation predictions: [Images/runs_YOLO11/classify/train7/val_batch1_pred.jpg](Images/runs_YOLO11/classify/train7/val_batch1_pred.jpg)

3. Test Results

- After evaluating multiple configurations, `ResNet-50 (Frozen Strategy)` was selected as the production gateway due to the best trade-off between high accuracy, computational stability, and low latency.
- Reported test accuracy for the chosen configuration: 0.9883.

General result images:

- [Images/Result.png](Images/Result.png)

4. Important repository files

- Trained ResNet-50 model (frozen): [Model/Invoice_Classification.pth](Model/Invoice_Classification.pth)
- Unfreeze experiment checkpoint: [Model/Last_Unfreez_Invoice_Classification.pth](Model/Last_Unfreez_Invoice_Classification.pth)
- YOLO train weights: [Images/runs_YOLO11/classify/train7/weights/best.pt](Images/runs_YOLO11/classify/train7/weights/best.pt)

5. Quick start

Requirements: Python 3.8+, PyTorch (CPU or CUDA), Pillow, torchvision.

Example: load the ResNet-50 model and run a prediction

```python
import torch
from PIL import Image
from torchvision import transforms

# Adjust these paths if needed
model_path = 'Model/Invoice_Classification.pth'
img_path = 'Images/runs_YOLO11/classify/train7/val_batch1_pred.jpg'

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)

model = torch.load(model_path, map_location='cpu')
model.eval()
with torch.no_grad():
    out = model(input_tensor)
    pred = out.argmax(dim=1).item()
    print('Predicted class id:', pred)
```

6. Notes & recommendations

- `ResNet-50 (Frozen)` was chosen for production for its balance of accuracy and inference latency. If you have ample compute, try `full fine-tuning` to squeeze additional accuracy.
- The `Images` directory contains visual examples, confusion matrices, and training snapshots used in the report.

7. References

- ReceiptSense dataset
- Relevant Kaggle datasets (Invoices / Menus)
- [Images/Reference.png](Images/Reference.png)

---

8. Author

- Developed by: Omar Hafez Khalil
- GitHub: [OmarHKhalil](https://github.com/OmarHKhalil)
- LinkedIn: [Omar Khalil](https://www.linkedin.com/in/omar-khalil-55a674281)
