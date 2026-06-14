# Model Card - Mushroom EfficientNet-B3 Classifier

## Model Overview

| Field | Value |
|-------|-------|
| **Architecture** | EfficientNet-B3 (pretrained on ImageNet) |
| **Framework** | PyTorch 2.x |
| **Task** | 30-class mushroom species image classification |
| **Input** | RGB image, 224×224 px, normalized with ImageNet statistics |
| **Output** | Softmax probability distribution over 30 species |
| **Final validation accuracy** | **83%** (600-sample val set) |
| **Macro F1-score** | 0.83 |

## Training

### Two-phase transfer learning

| Phase | Epochs | Layers trained | Optimizer | LR |
|-------|--------|---------------|-----------|-----|
| 1 - Head only | 5 | Classification head only | Adam | 1e-3 |
| 2 - Fine-tuning | 5 | Last 3 backbone blocks + head | Adam | 1e-4 |

- **Loss function**: Cross-Entropy
- **Batch size**: 32
- **Hardware**: Kaggle T4 GPU (~10 min total)

### Classification head

```
EfficientNet-B3 backbone (frozen / partially unfrozen)
    └── Dropout(p=0.3)
    └── Linear(1536 → 30)  [activation: Softmax at inference]
```

### Hyperparameters

| Parameter | Phase 1 (head only) | Phase 2 (fine-tuning) |
|-----------|--------------------|-----------------------|
| **Optimizer** | Adam | Adam |
| **Learning rate** | 1e-3 | 1e-4 |
| **Loss function** | CrossEntropyLoss | CrossEntropyLoss |
| **Performance metric** | Accuracy | Accuracy |
| **Epochs** | 5 | 5 |
| **Batch size** | 32 | 32 |
| **Validation** | val set (600 images) | val set (600 images) |
| **Dropout** | 0.3 | 0.3 |
| **Layers trained** | Head only | Last 3 backbone blocks + head |

## Performance

### Per-class highlights (validation set, 20 samples/class)

| Species | F1 |
|---------|----|
| Lycoperdon perlatum | 0.97 |
| Mycena leaiana | 0.95 |
| Hericium coralloides | 0.93 |
| Tricholomopsis rutilans | 0.67 |
| Boletus edulis | 0.62 |

### Known confusions

- **Parmelia sulcata ↔ Xanthoria parietina**: both foliose lichens with similar grey-green colouration.
- **Cerioporus squamosus ↔ Fomitopsis pinicola**: both bracket fungi with overlapping colour ranges.

## Intended Use

- Biodiversity monitoring and citizen science.
- Educational tool for identifying common mushroom species.
- Research baseline for fine-grained mushroom classification.

## Out-of-Scope Use

- Clinical or medical diagnosis.
- Solo determination of mushroom edibility for consumption - **always consult a professional mycologist**.

## Safety & Security Considerations

1. **Confidence threshold**: predictions below ~60% softmax confidence should trigger a warning rather than a species name. Never display a high-confidence prediction on an ambiguous image.
2. **Top-3 predictions**: display the three most likely species with their probabilities instead of a single answer, consistent with best practices from *Fungi Recognition: A Practical Use Case* (Sulc & Matas, 2019).
3. **Disclaimer**: the application must include a visible notice that AI-assisted identification does not replace expert mycological validation, especially before consumption.
4. **Model robustness**: the model has not been adversarially tested. It may be susceptible to distribution shift (unusual lighting, partial views, immature specimens).
5. **Scope limitation**: trained on 30 species; it will silently force any input into one of those 30 classes. Images of out-of-scope species will be misclassified with high confidence.

## Limitations

- Trained on a small subset (1,500 images, 30 species) of a 100+ species dataset.
- Accuracy varies across species (range 62%–97%).
- No test-time augmentation or ensemble - a production system should use both.
