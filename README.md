# IMDb Sentiment Analysis with PyTorch and Transformers

This project demonstrates a full sentiment analysis pipeline using the IMDb dataset, HuggingFace Transformers (DistilBERT), and PyTorch. It includes training, evaluation, inference on custom and unsupervised data, and proper code structure for reproducibility and portfolio visibility.

---

## 📂 Project Structure
llm-sentiment-pipeline/
├── scripts/
│ ├── 01_train.py # Train DistilBERT on IMDb
│ ├── 02_eval_report.py # Classification report on test set
│ ├── 03_infer_single.py # Predict sentiment for a single text
│ └── 04_infer_unsupervised.py # Run model on IMDb unsupervised split
├── model/ # Saved tokenizer & model (after training)
└── README.md # This file

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch datasets transformers tqdm scikit-learn
```
```bash
2. Train Model
python scripts/01_train.py
```
3. Evaluate Performance
```bash
python scripts/02_eval_report.py
```
4. Predict a Custom Review
```bash
python scripts/03_infer_single.py
```
5. Run on Unsupervised IMDb Reviews
```bash
python scripts/04_infer_unsupervised.py
```

🧪 Model Info
Base model: distilbert-base-uncased

Dataset: IMDb from Ηuggingface Datasets

Task: Binary Sentiment Classification (Positive / Negative)

📌 Notes
This project is intended for educational & portfolio purposes.

Can be extended to include pseudo-labeling, training metrics visualization, or fine-tuning larger LLMs.
