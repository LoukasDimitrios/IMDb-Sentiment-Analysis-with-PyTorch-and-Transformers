# IMDb Sentiment Analysis with PyTorch and Transformers

This project demonstrates a full sentiment analysis pipeline using the IMDb dataset, HuggingFace Transformers (DistilBERT), and PyTorch. It includes training, evaluation, inference on both labeled and unlabeled data, and is structured to be beginner-friendly and easy to extend.

---

## 📂 Project Structure
```text
IMDb-Sentiment-Analysis-with-PyTorch-and-Transformers/
├── scripts/
│   ├── 01_train.py              # Train DistilBERT on IMDb
│   ├── 02_eval_report.py        # Classification report on test set
│   ├── 03_infer_single.py       # Predict sentiment for a single text
│   └── 04_infer_unsupervised.py # Run model on IMDb unsupervised split
├── model/                       # Saved tokenizer & model (after training)
└── README.md                    # This file
```
---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch datasets transformers tqdm scikit-learn
```
### 2. Train Model
```bash
python scripts/01_train.py
```
### 3. Evaluate Performance
```bash
python scripts/02_eval_report.py
```
### 4. Predict a Custom Review
```bash
python scripts/03_infer_single.py
```
### 5. Run on Unsupervised IMDb Reviews
```bash
python scripts/04_infer_unsupervised.py
```

## 🧪 Model Info
  - Base model: distilbert-base-uncased

  - Dataset: IMDb from HuggingFace Datasets

  - Task: Binary Sentiment Classification (Positive / Negative)

  - link: https://huggingface.co/distilbert-base-uncased

## 📌 Notes
Potential extensions:
  - pseudo-labeling
    
  - training metrics visualization
    
  - fine-tuning larger LLMs
