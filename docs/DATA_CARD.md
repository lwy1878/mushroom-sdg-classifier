# Data Card - Mushroom Species Recognition Dataset

## Dataset Overview

| Field | Value |
|-------|-------|
| **Name** | Mushroom Species Recognition (mushroom1) |
| **Source** | [Kaggle - zlatan599/mushroom1](https://www.kaggle.com/datasets/zlatan599/mushroom1) |
| **Task** | Fine-grained image classification |
| **Species covered** | 100+ mushroom / lichen / fungal species |
| **Image format** | JPEG, various resolutions |
| **Splits** | train.csv / val.csv / test.csv |

## Content

Each CSV row contains:
- `image_path`: relative path to the image file
- `label`: species name (Latin binomial, e.g. *Amanita muscaria*)

## Subset Used in This Project

To enable rapid prototyping on a Kaggle T4 GPU:

| Split | Classes | Images per class | Total |
|-------|---------|-----------------|-------|
| Train | 30 (top by frequency) | 50 | 1,500 |
| Validation | 30 (same as train) | 20 | 600 |

Only species present in the training split are kept in validation to avoid label-encoding errors.

## Preprocessing

| Step | Training | Validation |
|------|----------|------------|
| Resize | 224×224 | 224×224 |
| Random horizontal flip | ✓ | — |
| Random rotation ±15° | ✓ | — |
| Colour jitter (brightness/contrast ±0.2) | ✓ | — |
| Normalize (ImageNet mean/std) | ✓ | ✓ |

## Limitations & Biases

- Images were collected from internet sources; lighting and background conditions vary widely.
- The 30-species subset under-represents rare and visually similar species.
- Geographic and seasonal diversity of the original collection is unknown.
- **No toxicity labels** are provided. Do not use model predictions alone to determine edibility.

## SDG Alignment

| SDG | Relevance |
|-----|-----------|
| SDG 3 - Good Health & Well-being | Reducing accidental poisoning from misidentified mushrooms |
| SDG 15 - Life on Land | Supporting biodiversity monitoring and conservation of fungal species |
