from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

# --- 1.Download and prepare the model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5) # 5 فئات
    model.load_state_dict(torch.load('Model/best_resnet50_menu_classification.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
class_names = ['Drink', 'Food', 'Interior', 'Menu', 'Outside']

# --- 2.Image processing settings ---
preprocess = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Image reading and conversion
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Processing and prediction
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    return {
        "prediction": class_names[predicted.item()],
        "confidence": float(confidence.item()),
        "all_probs": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    }