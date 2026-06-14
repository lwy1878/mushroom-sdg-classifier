import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import models, transforms
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

num_classes = len(le.classes_)

# Build EfficientNet-B3 model
model = models.efficientnet_b3(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1536, num_classes)
)

# Load trained weights
model.load_state_dict(torch.load("mushroom_efficientnet_b3.pth", map_location=device))
model.to(device)
model.eval()

# Same preprocessing as validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():
    return {"message": "Mushroom EfficientNet-B3 API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        species = le.classes_[idx.item()]
        confidence = round(prob.item() * 100, 2)
        results.append({
            "species": species,
            "confidence": confidence
        })

    warning = None
    if results[0]["confidence"] < 60:
        warning = "Low confidence prediction. Please consult a professional mycologist."

    return {
        "top_3_predictions": results,
        "warning": warning,
        "disclaimer": "This AI result should not be used alone to determine mushroom edibility."
    }