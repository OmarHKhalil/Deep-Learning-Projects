# 🍽️ Smart Menu Classifier System

An end-to-end deep learning project for **restaurant-related image classification** using **PyTorch**, **ResNet50**, **FastAPI**, and **Streamlit**.

The system classifies restaurant images into **5 categories**:

* **Drink**
* **Food**
* **Interior**
* **Menu**
* **Outside**

This project includes:

* **Model training and evaluation** using PyTorch
* **A FastAPI backend** for model inference
* **A Streamlit frontend** for interactive image upload and prediction
* **Batch prediction on folders of images**
* **Performance evaluation** with accuracy, precision, recall, F1-score, and confusion matrix

---

# 📌 Project Overview

The goal of this project is to build an intelligent image classification system for restaurant-related content.
Given an uploaded image, the system predicts whether it belongs to one of the following categories:

* 🍹 **Drink**
* 🍔 **Food**
* 🪑 **Interior**
* 📄 **Menu**
* 🏠 **Outside**

The project was designed as a **complete ML application pipeline**, not just a training notebook.
It includes:

1. **Model development and training**
2. **Inference API using FastAPI**
3. **Interactive web interface using Streamlit**

---

# 🧠 Classes

| Class      | Description                       |
| ---------- | --------------------------------- |
| `Drink`    | Beverage-related images           |
| `Food`     | Food dishes and meals             |
| `Interior` | Indoor restaurant environment     |
| `Menu`     | Menu photos or menu boards        |
| `Outside`  | Exterior restaurant/building view |

---

# 🏗️ Project Architecture

The project consists of **three main components**:

## 1) Deep Learning Model

* Backbone: **ResNet50**
* Framework: **PyTorch**
* Input image size: **448 × 448**
* Output classes: **5**

## 2) FastAPI Backend

The backend loads the trained model and exposes a prediction endpoint:

* `POST /predict`

It receives an image file and returns:

* predicted class
* confidence score
* probability distribution across all classes

## 3) Streamlit Frontend

A user-friendly web interface where users can:

* Upload an image
* Send it to the FastAPI backend
* View the predicted category
* View confidence score
* Visualize class probabilities in a Plotly bar chart

---

# 📁 Project Structure

The project is currently organized as follows:

```bash
Smart-Menu-Classifier/
│
├── Menu_Classification.mp4
├── 
├── Dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── env/
│
├── Model/
│   └── best_resnet50_menu_classification.pth
│
├── app.py                  # FastAPI backend
├── app_ui.py               # Streamlit frontend
├── Menu_Classification.ipynb
├── README.md
└── requirements.txt
```
---

# 🖼️ System Workflow

The full system works as follows:

1. The user uploads an image from the **Streamlit UI**
2. Streamlit sends the image to the **FastAPI backend**
3. FastAPI preprocesses the image
4. The trained **ResNet50** model predicts the class
5. FastAPI returns:

   * top predicted class
   * confidence score
   * probabilities for all categories
6. Streamlit displays the result and plots the class distribution

---

# 🏋️ Model Training

The model was trained using **ResNet50** for a 5-class restaurant image classification task.

## Model setup

```python
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
```

## Training settings

* **Epochs:** 20
* **Batch size:** 16
* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Loss function:** CrossEntropyLoss
* **Early stopping patience:** 5
* **Image size:** 448 × 448

---

# 📂 Dataset Structure

The dataset is expected to follow this structure:

```bash
Dataset/
│
├── train/
│   ├── drink/
│   ├── food/
│   ├── interior/
│   ├── menu/
│   └── outside/
│
├── valid/
│   ├── drink/
│   ├── food/
│   ├── interior/
│   ├── menu/
│   └── outside/
│
└── test/
    ├── drink/
    ├── food/
    ├── interior/
    ├── menu/
    └── outside/
```

---

# 📊 Dataset Size

* **Training:** 6000 images
* **Validation:** 1000 images
* **Testing:** 1000 images

---

# ⚙️ Data Preprocessing

## Training transforms

```python
transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

## Validation / Test / Inference transforms

```python
transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

---

# 🚀 Model Performance

## Final Evaluation

| Dataset    | Loss       | Accuracy   |
| ---------- | ---------- | ---------- |
| Training   | **0.1872** | **93.77%** |
| Validation | **0.2409** | **91.60%** |
| Testing    | **0.2281** | **92.20%** |

## Additional Metrics

* **Precision:** `0.9232`
* **Recall:** `0.9219`
* **F1 Score:** `0.9222`

---

# 🧪 Test Set Classification Report

```text
              precision    recall  f1-score   support

Drink            0.95      0.93      0.94       200
Food             0.95      0.95      0.95       202
Interior         0.84      0.90      0.87       200
Menu             0.95      0.96      0.96       200
Outside          0.93      0.87      0.90       198

accuracy                              0.92      1000
macro avg         0.92      0.92      0.92      1000
weighted avg      0.92      0.92      0.92      1000
```

---

# 🔍 Confusion Matrix

```text
[[186   4   5   4   1]
 [  3 191   5   0   3]
 [  3   6 180   3   8]
 [  1   0   5 192   2]
 [  2   1  19   3 173]]
```

The model performs strongly across most categories, with some confusion between **Interior** and **Outside**, which is expected because some restaurant scenes share similar visual cues.

---

# ⚡ FastAPI Backend

The backend is implemented in **`app.py`** using FastAPI.

## Main endpoint

### `POST /predict`

Accepts an uploaded image file and returns a JSON response containing:

* top prediction
* confidence score
* probabilities for all classes

## Example response

```json
{
  "prediction": "Food",
  "confidence": 0.9642,
  "all_probs": {
    "Drink": 0.0121,
    "Food": 0.9642,
    "Interior": 0.0084,
    "Menu": 0.0105,
    "Outside": 0.0048
  }
}
```

---

# 🧪 FastAPI Inference Logic

The API loads the model once when the application starts:

```python
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load('best_resnet50_menu_classification.pth', map_location=device))
    model.to(device)
    model.eval()
    return model
```

It then preprocesses the uploaded image and returns predictions through the `/predict` endpoint.

---

# 🖥️ Streamlit Frontend

The frontend is implemented in **`app_ui.py`**.

It provides an interactive interface where the user can:

* upload an image
* preview it
* click **Analyze Image**
* receive the predicted category and confidence score
* visualize probability distribution with a Plotly chart

## Frontend features

* Clean two-column layout
* Styled prediction card
* Sidebar project information
* Upload preview
* Probability visualization with Plotly
* Error handling if the FastAPI backend is not running

---

# 🔗 Frontend–Backend Communication

The Streamlit app sends uploaded images to FastAPI using:

```python
response = requests.post("http://127.0.0.1:8000/predict", files=files)
```

So the backend must be running before using the frontend.

---

# ▶️ How to Run the Project

## 1) Clone the repository

```bash
git clone https://github.com/OmarHKhalil/Deep-Learning-Projects.git
cd smart-menu-classifier
```

---

## 2) Install dependencies

```bash
pip install -r requirements.txt
```
---

## 3) Place the trained model weights

Put the trained model file inside the `Model/` folder:

```bash
Model/best_resnet50_menu_classification.pth
```

---

## 4) Update the model path inside `app.py`

If your weights are inside the `Model/` folder, make sure the loading line points to the correct path.

Example:

```python
model.load_state_dict(
    torch.load("Model/best_resnet50_menu_classification.pth", map_location=device)
)
```

---

## 5) Run FastAPI backend

From the project root, run:

```bash
uvicorn app:app --reload
```

FastAPI will start on:

```bash
http://127.0.0.1:8000
```

---

## 6) Run Streamlit frontend

In a new terminal, run:

```bash
streamlit run app_ui.py
```

---

## 7) Use the application

1. Open the Streamlit page in your browser
2. Upload a restaurant-related image
3. Click **Analyze Image**
4. View:

   * predicted category
   * confidence score
   * probability chart

---

# 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **FastAPI**
* **Uvicorn**
* **Streamlit**
* **Plotly**
* **TorchMetrics**
* **Pillow**
* **Scikit-learn**

---

## Author

* **Developed by:** Omar Hafez Khalil
* **GitHub:** [OmarHKhalil](https://github.com/OmarHKhalil)
* **LinkedIn:** [Omar Khalil](https://www.linkedin.com/in/omar-khalil-55a674281)
