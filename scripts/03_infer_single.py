# file: scripts/03_infer_single.py

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_dir = "model/"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Example input
text = "This movie was absolutely stunning!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

# Output label
label = "positive" if pred == 1 else "negative"
print(f"Prediction: {label}")
