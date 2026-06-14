# Deployment Guide

## Mushroom Species Recognition System

This document describes the deployment process of the Mushroom Species Recognition System using Microsoft Azure Virtual Machine and FastAPI.

---

## 1. System Architecture

```text
User
  ↓
Web Browser
  ↓
FastAPI Web Application
  ↓
EfficientNet-B3 Model
  ↓
Prediction Result
```

### Technologies Used

* Python 3.12
* FastAPI
* PyTorch
* EfficientNet-B3
* Microsoft Azure Virtual Machine
* Ubuntu Linux

---

## 2. Azure Virtual Machine Setup

### Create Azure VM

Configuration:

* Operating System: Ubuntu 24.04 LTS
* Region: East US
* Authentication: Password Authentication
* Public IP: Enabled

Open inbound ports:

* SSH (22)
* HTTP (80)
* FastAPI (8000)

---

## 3. Connect to Azure VM

Use SSH to access the virtual machine:

```bash
ssh username@public-ip-address
```

Example:

```bash
ssh lwy1878@<PUBLIC_IP>
```

---

## 4. Clone Project Repository

```bash
git clone <repository-url>
cd mushroom-sdg-classifier
```

---

## 5. Create Python Environment

Install venv package:

```bash
sudo apt update
sudo apt install python3.12-venv -y
```

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 6. Install Dependencies

```bash
pip install -r requirements.txt
```

Verify installation:

```bash
pip list
```

Important packages:

* fastapi
* uvicorn
* torch
* torchvision
* scikit-learn
* pillow

---

## 7. Upload Model Files

Required files:

```text
mushroom_efficientnet_b3.pth
label_encoder.pkl
```

Upload using SCP:

```bash
scp mushroom_efficientnet_b3.pth username@public-ip:/home/username/project-folder/
scp label_encoder.pkl username@public-ip:/home/username/project-folder/
```

---

## 8. Run the Application

Activate virtual environment:

```bash
source venv/bin/activate
```

Start FastAPI server:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Expected output:

```text
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## 9. Access the Web Application

Open:

```text
http://<PUBLIC_IP>:8000
```

Swagger API documentation:

```text
http://<PUBLIC_IP>:8000/docs
```

---

## 10. Features

The deployed system supports:

* Image upload
* Mushroom image preview
* Top-3 species prediction
* Confidence score visualization
* Safety disclaimer display
* Real-time inference through FastAPI

---

## 11. Disclaimer

This AI system is designed for educational and research purposes only.

Predictions should not be used as the sole basis for determining mushroom edibility. Always consult qualified experts before consuming wild mushrooms.

```
```
