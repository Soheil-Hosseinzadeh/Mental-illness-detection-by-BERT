import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
import os

# -------------------------------
# Load CSV
# -------------------------------
csv_file = "suicidal.csv"
df = pd.read_csv(csv_file)
dataset = Dataset.from_pandas(df)

# Train/val/test split 70/15/15
train_val_test = dataset.train_test_split(test_size=0.30, random_state=42)
val_test = train_val_test["test"].train_test_split(test_size=0.50, random_state=42)
train_dataset, val_dataset, test_dataset = (
    train_val_test["train"],
    val_test["train"],
    val_test["test"],
)

# Tokenizer
model_name = "mental/mental-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "not_suicidal", 1: "suicidal"},
    label2id={"not_suicidal": 0, "suicidal": 1},
)


# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}


# Training arguments
training_args = TrainingArguments(
    output_dir="./results_suicidal",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs_suicidal",
    logging_strategy="epoch",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Final evaluation
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

report = classification_report(
    predictions.label_ids,
    preds,
    target_names=["not_suicidal", "suicidal"],
    output_dict=True,
)
os.makedirs("./results_suicidal", exist_ok=True)
with open("./results_suicidal/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

trainer.save_model("./results_suicidal/model")
tokenizer.save_pretrained("./results_suicidal/model")
