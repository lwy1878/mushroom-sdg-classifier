# Mushroom Species Recognition

**SDG 3 - Good Health & Well-being | SDG 15 - Life on Land**

An image classifier that identifies mushroom species from photos using transfer learning (EfficientNet-B3). Achieves **83% validation accuracy** on 30 species in ~10 minutes on a T4 GPU.

**Team:** Paul Arbez (114012021) · Baptiste Caulier (114012019) · 呂維洋 (111370218)

---

## Quick Start - Run on Kaggle (recommended)

The easiest way to reproduce results is directly on Kaggle where the dataset is already mounted:

1. Open the notebook: [kaggle.com/code/paularbez/mushroom-species-recognition](https://www.kaggle.com/code/paularbez/mushroom-species-recognition)
2. Click **Copy & Edit**
3. Make sure the dataset [`zlatan599/mushroom1`](https://www.kaggle.com/datasets/zlatan599/mushroom1) is attached (it is by default in the original notebook)
4. Set accelerator to **GPU T4 x1**
5. Click **Run All**

---

## Local Setup

### 1. Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU works but is slow)

### 2. Clone & install dependencies

```bash
git clone https://github.com/<your-org>/mushroom-sdg-classifier.git
cd mushroom-sdg-classifier
pip install -r requirements.txt
```

### 3. Download the dataset

Download [`zlatan599/mushroom1`](https://www.kaggle.com/datasets/zlatan599/mushroom1) from Kaggle and unzip it:

```bash
# install kaggle CLI if needed
pip install kaggle
kaggle datasets download -d zlatan599/mushroom1 --unzip -p data/
```

The expected layout is:
```
data/
  train.csv
  val.csv
  test.csv
  merged_dataset/
    Amanita muscaria/
      *.jpg
    ...
```

### 4. Train

```bash
python src/train.py --data-dir data/
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | required | Root of the mushroom1 dataset |
| `--n-classes` | 30 | Number of top species to keep |
| `--train-samples` | 50 | Max training images per class |
| `--val-samples` | 20 | Max validation images per class |
| `--epochs-head` | 5 | Phase 1 epochs (head only) |
| `--epochs-ft` | 5 | Phase 2 epochs (fine-tuning) |
| `--output-dir` | outputs/ | Where to save model & plots |

### 5. Outputs

After training, `outputs/` contains:

```
outputs/
  mushroom_efficientnet_b3.pth   # model weights
  label_encoder.pkl              # LabelEncoder (maps int → species name)
  metrics.json                   # final accuracy + full history
  training_curves.png            # loss & accuracy plots
  confusion_matrix.png           # top-10 class confusion matrix
```

---

## Project Structure

```
mushroom-sdg-classifier/
├── notebooks/
│   └── mushroom-species-recognition.ipynb   # Kaggle notebook (runnable as-is)
├── src/
│   └── train.py                             # CLI training script (local use)
├── docs/
│   ├── DATA_CARD.md
│   └── MODEL_CARD.md
├── requirements.txt
└── README.md
```

---

## Results

| Phase | Epochs | Val Accuracy |
|-------|--------|-------------|
| Head only | 1–5 | 56% → 67.5% |
| Fine-tuning | 6–10 | 69% → **83%** |

Macro F1: **0.83** across 30 species (600-image validation set).

---

## Security & Safety

Misidentifying a toxic mushroom as edible can be fatal. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md#safety--security-considerations) for the full discussion. Key principles applied:

- Display **top-3 predictions** with confidence scores, not a single answer.
- Include a **confidence threshold**: warn users when the model is uncertain.
- Always show a **disclaimer** advising users to consult a professional mycologist before consuming any wild mushroom.

Reference paper: *Fungi Recognition: A Practical Use Case* - Sulc & Matas, 2019 ([arXiv:1806.05712](https://arxiv.org/abs/1806.05712)).

---

## Documentation

- [Data Card](docs/DATA_CARD.md) - dataset provenance, splits, preprocessing, biases
- [Model Card](docs/MODEL_CARD.md) - architecture, training, performance, safety
