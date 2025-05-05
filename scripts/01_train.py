from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.optim import AdamW

# Load the IMDb dataset
print("Loading IMDb dataset...")
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoaders
train_loader = DataLoader(tokenized["train"], batch_size=8, shuffle=True)
eval_loader = DataLoader(tokenized["test"], batch_size=8)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * 3
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Training
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        if 'label' in batch:
            batch['labels'] = batch.pop('label')

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

    acc = correct / total
    print(f"Train Loss: {train_loss / len(train_loader):.4f} | Accuracy: {acc:.4f}")

    # Evaluation per epoch
    model.eval()
    eval_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if 'label' in batch:
                batch['labels'] = batch.pop('label')

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    acc = correct / total
    print(f"Eval Loss: {eval_loss / len(eval_loader):.4f} | Accuracy: {acc:.4f}\n")

# Save model and tokenizer
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")