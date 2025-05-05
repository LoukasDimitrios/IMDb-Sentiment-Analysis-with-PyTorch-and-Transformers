# file: scripts/02_eval_report.py

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm

# Load model & tokenizer
model_dir = "model/"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load IMDb dataset for evaluation
dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset["test"].map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_loader = DataLoader(tokenized, batch_size=8)

# Run inference on test set
y_true = []
y_pred = []
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        if 'label' in batch:
            batch['labels'] = batch.pop('label')

        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)

        y_true.extend(batch["labels"].cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))
