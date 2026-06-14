"""
Mushroom Species Recognition — local training script.
Mirrors the Kaggle notebook exactly but accepts a local data directory.

Usage:
    python src/train.py --data-dir /path/to/mushroom1
"""

import os
import json
import argparse
import pickle
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Root of the mushroom1 dataset")
    p.add_argument("--n-classes", type=int, default=30)
    p.add_argument("--train-samples", type=int, default=50)
    p.add_argument("--val-samples", type=int, default=20)
    p.add_argument("--epochs-head", type=int, default=5)
    p.add_argument("--epochs-ft", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def sample_dataset(df, n_classes, samples_per_class):
    top_classes = df["label"].value_counts().head(n_classes).index
    return df[df["label"].isin(top_classes)].groupby("label").head(samples_per_class).reset_index(drop=True)


class MushroomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df["image_path"][idx]).convert("RGB")
        label = self.df["label_enc"][idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes, device):
    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct = 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(train_accs, label="Train")
    ax2.plot(val_accs, label="Val")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion(all_labels, all_preds, le, out_dir):
    top10_idx = [i for i, _ in Counter(all_labels).most_common(10)]
    mask = [i for i, l in enumerate(all_labels) if l in top10_idx]
    cm = confusion_matrix(
        [all_labels[i] for i in mask],
        [all_preds[i] for i in mask],
        labels=top10_idx,
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=le.classes_[top10_idx],
        yticklabels=le.classes_[top10_idx],
    )
    plt.title("Confusion Matrix (top 10 classes)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(args.data_dir, "val.csv"))

    # Remap paths that still point to /kaggle/working/
    for df in (train_df, val_df):
        df["image_path"] = df["image_path"].str.replace(
            "/kaggle/working/", args.data_dir + "/", regex=False
        )

    train_small = sample_dataset(train_df, args.n_classes, args.train_samples)
    train_classes = set(train_small["label"].unique())
    val_small = (
        val_df[val_df["label"].isin(train_classes)]
        .groupby("label").head(args.val_samples)
        .reset_index(drop=True)
    )

    le = LabelEncoder()
    train_small["label_enc"] = le.fit_transform(train_small["label"])
    val_small["label_enc"]   = le.transform(val_small["label"])

    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    num_classes = len(le.classes_)
    print(f"Classes: {num_classes} | Train: {len(train_small)} | Val: {len(val_small)}")

    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(
        MushroomDataset(train_small, train_tf),
        batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        MushroomDataset(val_small, val_tf),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Phase 1: head only ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    print("\n── Phase 1: head only ──")
    for epoch in range(args.epochs_head):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"  Epoch {epoch+1}/{args.epochs_head} | "
              f"Train {ta:.3f} | Val {va:.3f}")

    # ── Phase 2: fine-tuning ──────────────────────────────────────────────────
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    print("\n── Phase 2: fine-tuning ──")
    for epoch in range(args.epochs_ft):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        print(f"  FT Epoch {epoch+1}/{args.epochs_ft} | "
              f"Train {ta:.3f} | Val {va:.3f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(args.output_dir, "mushroom_efficientnet_b3.pth"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "final_val_acc": history["val_acc"][-1],
                "final_train_acc": history["train_acc"][-1],
                "history": history,
            },
            f, indent=2,
        )

    plot_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
        args.output_dir,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    plot_confusion(all_labels, all_preds, le, args.output_dir)
    print("\n" + classification_report(all_labels, all_preds, target_names=le.classes_))
    print(f"\nOutputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
