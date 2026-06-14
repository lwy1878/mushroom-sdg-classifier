import io
import pickle
import warnings

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import models, transforms

warnings.filterwarnings("ignore")

app = FastAPI(title="Mushroom Species Recognition")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

NUM_CLASSES = len(le.classes_)

model = models.efficientnet_b3(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)

state_dict = torch.load("mushroom_efficientnet_b3.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mushroom Species Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 950px;
            margin: 50px auto;
            background: white;
            padding: 40px;
            border-radius: 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            text-align: center;
        }

        h1 {
            color: #2e7d32;
            font-size: 38px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .upload-box {
            border: 2px dashed #81c784;
            padding: 30px;
            border-radius: 18px;
            background: #f9fff9;
            margin-bottom: 25px;
            text-align: center;
        }

        input[type=file] {
            margin: 15px 0;
        }

        #preview {
            width: 350px;
            height: 350px;
            object-fit: cover;
            display: none;
            margin: 25px auto;
            border-radius: 18px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 8px 20px rgba(0,0,0,0.18);
        }

        button {
            background: #2e7d32;
            color: white;
            border: none;
            padding: 16px 40px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
        }

        button:hover {
            background: #1b5e20;
        }

        button:disabled {
            background: #9e9e9e;
            cursor: not-allowed;
        }

        .result {
            margin-top: 30px;
            text-align: left;
            display: none;
        }

        .card {
            background: #f1f8e9;
            padding: 18px;
            border-radius: 14px;
            margin: 12px 0;
            border-left: 6px solid #43a047;
        }

        .species {
            font-size: 22px;
            font-weight: bold;
            color: #2e7d32;
        }

        .confidence {
            color: #555;
            margin-top: 8px;
        }

        .rank-badge {
            display: inline-block;
            background: #2e7d32;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 13px;
            margin-right: 10px;
        }

        .progress {
            width: 100%;
            height: 12px;
            background: #dcedc8;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #81c784, #2e7d32);
            border-radius: 999px;
        }

        .warning {
            margin-top: 20px;
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            font-weight: 500;
            line-height: 1.5;
        }

        .footer {
            margin-top: 30px;
            color: #777;
            font-size: 14px;
        }

        .loading {
            display: none;
            color: #2e7d32;
            margin-top: 15px;
            font-weight: bold;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>🍄 Mushroom Species Recognition</h1>
        <div class="subtitle">
            EfficientNet-B3 model deployed on Azure VM with FastAPI
        </div>

        <div class="upload-box">
            <h3>Upload a mushroom image</h3>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage()">
            <br>
            <img id="preview">
            <br>
            <button id="predictBtn" onclick="predict()">Predict</button>
            <div id="loading" class="loading">Analyzing image...</div>
        </div>

        <div id="result" class="result">
            <h2>Prediction Result</h2>
            <div id="predictionCards"></div>

            <div class="warning">
                ⚠️ Disclaimer: This AI result should not be used alone to determine mushroom edibility.
                Always consult a professional mycologist before consuming any wild mushroom.
            </div>
        </div>

        <div class="footer">
            SDG 3: Good Health & Well-being | SDG 15: Life on Land
        </div>
    </div>

<script>
function previewImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const preview = document.getElementById("preview");
    const resultDiv = document.getElementById("result");

    resultDiv.style.display = "none";

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    } else {
        preview.style.display = "none";
    }
}

async function predict() {
    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const loading = document.getElementById("loading");
    const resultDiv = document.getElementById("result");
    const cardsDiv = document.getElementById("predictionCards");

    if (!fileInput.files.length) {
        alert("Please upload an image first.");
        return;
    }

    predictBtn.disabled = true;
    loading.style.display = "block";
    cardsDiv.innerHTML = "";
    resultDiv.style.display = "none";

    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }

        const data = await response.json();
        const predictions = data.top_3_predictions || [];

        if (predictions.length > 0) {
            predictions.forEach((item, index) => {
                const species = item.species || "Unknown";
                const confidence = Number(item.confidence || 0);

                cardsDiv.innerHTML += `
                    <div class="card">
                        <div class="species">
                            <span class="rank-badge">Top ${index + 1}</span>
                            ${species}
                        </div>

                        <div class="confidence">
                            Confidence: ${confidence.toFixed(2)}%
                        </div>

                        <div class="progress">
                            <div class="progress-bar"
                                 style="width:${Math.min(confidence, 100)}%">
                            </div>
                        </div>
                    </div>
                `;
            });

            if (Number(predictions[0].confidence) < 60) {
                cardsDiv.innerHTML += `
                    <div class="warning">
                        ⚠️ Low confidence prediction. Please consult a professional mycologist.
                    </div>
                `;
            }
        } else {
            cardsDiv.innerHTML = `
                <div class="card">
                    <div class="species">No prediction returned.</div>
                </div>
            `;
        }

        resultDiv.style.display = "block";

    } catch (error) {
        cardsDiv.innerHTML = `
            <div class="warning">
                Prediction failed. Please try another image.
            </div>
        `;
        resultDiv.style.display = "block";
    } finally {
        predictBtn.disabled = false;
        loading.style.display = "none";
    }
}
</script>
</body>
</html>
"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=3)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "species": str(le.classes_[idx.item()]),
            "confidence": round(float(prob.item()) * 100, 2)
        })

    return {
        "top_3_predictions": results,
        "disclaimer": "This AI result should not be used alone to determine mushroom edibility."
    }
