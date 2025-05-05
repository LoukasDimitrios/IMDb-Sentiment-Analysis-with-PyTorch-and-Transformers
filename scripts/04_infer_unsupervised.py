# file: scripts/04_infer_unsupervised.py

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

# Load tokenizer & model
model_dir = "model/"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load IMDb dataset (unsupervised split)
dataset = load_dataset("imdb")
unsup_dataset = dataset["unsupervised"].remove_columns("label")

# Tokenization
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

unsup_dataset = unsup_dataset.map(tokenize, batched=True)
unsup_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
unsup_loader = DataLoader(unsup_dataset, batch_size=8)

# Run inference on unsupervised set
predictions = []
with torch.no_grad():
    for batch in tqdm(unsup_loader, desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

# Print sample predictions
for i in range(10):
    text = dataset["unsupervised"][i]["text"]
    pred = predictions[i]
    label = "positive" if pred == 1 else "negative"
    print(f"\n--- Review {i+1} ---")
    print(f"Predicted: {label}")
    print(text[:500], "...")
