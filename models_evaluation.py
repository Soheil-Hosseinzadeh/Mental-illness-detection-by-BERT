import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


EVALUATION_TARGETS = {
    "suicidal": {
        "csv_path": "Suicide_Detection.csv",
        "model_path": "results_suicidal/model",
    },
    "depression": {
        "csv_path": "depression.csv",
        "model_path": "results_depression/model",
    },
}
MAX_LENGTH = 256
BATCH_SIZE = 32

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()
    # 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_classes(texts, tokenizer, model, device, max_length=MAX_LENGTH, batch_size=BATCH_SIZE):
    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)   # (batch_size, num_classes)
            preds = torch.argmax(probs, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy().tolist())
    return all_preds, all_probs

def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    plt.figure()
    im = plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=8)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')


    classes = ['0', '1']
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)


    plt.xticks(np.arange(-.5, 2, 1), minor=True)
    plt.yticks(np.arange(-.5, 2, 1), minor=True)
    plt.grid(which='minor', color='white', linestyle='-', linewidth=1)
    plt.tick_params(which='minor', bottom=False, left=False)


    thr_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{int(cm[i,j])}", ha='center', va='center',
                     color="white" if cm[i,j] > thr_val else "#08306B", fontsize=12)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, target in EVALUATION_TARGETS.items():
        csv_path = target["csv_path"]
        model_path = target["model_path"]
        (X_train, _), (_, _), (X_test, y_test) = load_data(csv_path)
        print(f"\n{name.title()} train/test sizes: {len(X_train)}/{len(X_test)}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        tokenizer, model = load_model(model_path, device)
        y_pred, probs = predict_classes(X_test, tokenizer, model, device)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test: {acc:.4f}")
        print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        plot_confusion_matrix(cm, save_path=f"confusion_matrix_{name}.png")

if __name__ == "__main__":
    main()