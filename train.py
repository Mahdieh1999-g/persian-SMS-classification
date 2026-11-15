import os, re, json, random, unicodedata
import numpy as np
import pandas as pd
import torch, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "sms_examples.csv")
ALT_DATA_PATH = os.path.join(ROOT, "sms_examples_augmented.csv")
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED = 128, 8, 3, 2e-5, 42

def normalize_persian(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    text = text.replace("\u200c", " ")
    return re.sub(r"\\s+", " ", text).strip()

class SmsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, padding="max_length",
                                   max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def train_one_label(df, label_name, tokenizer, device):
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[label_name].astype(str))
    X = df["text"].tolist()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval)
    train_loader = DataLoader(SmsDataset(X_train, y_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SmsDataset(X_val, y_val, tokenizer), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SmsDataset(X_test, y_test, tokenizer), batch_size=BATCH_SIZE)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(encoder.classes_)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_f1, best_state = -1, None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += output.loss.item()
        model.eval()
        truth, preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                truth.extend(batch["labels"].tolist())
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                preds.extend(model(**inputs).logits.argmax(1).cpu().tolist())
        score = f1_score(truth, preds, average="macro", zero_division=0)
        print(f"{label_name} | epoch {epoch+1}/{EPOCHS} | loss={total_loss/max(1,len(train_loader)):.4f} | val_macro_f1={score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    truth, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            truth.extend(batch["labels"].tolist())
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            preds.extend(model(**inputs).logits.argmax(1).cpu().tolist())
    print(f"\\n=== TEST REPORT: {label_name} ===")
    print(classification_report(truth, preds, labels=list(range(len(encoder.classes_))),
                                target_names=[str(x) for x in encoder.classes_], zero_division=0))
    print(f"Accuracy: {accuracy_score(truth,preds):.4f} | Macro-F1: {f1_score(truth,preds,average='macro',zero_division=0):.4f}")
    out = os.path.join(MODEL_DIR, f"{label_name}_model")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    joblib.dump(encoder, os.path.join(out, "label_encoder.joblib"))
    with open(os.path.join(out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump([str(x) for x in encoder.classes_], f, ensure_ascii=False, indent=2)
    print("Saved:", out)

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    path = DATA_PATH if os.path.exists(DATA_PATH) else ALT_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError("Put sms_examples.csv in data/ or sms_examples_augmented.csv beside train.py")
    df = pd.read_csv(path).dropna(subset=["text","category","risk"]).copy()
    for col in ["text","category","risk"]: df[col] = df[col].astype(str).str.strip()
    df = df[df.text != ""].drop_duplicates(subset=["text","category","risk"])
    df["text"] = df["text"].map(normalize_persian)
    print("Dataset:", path, "| rows:", len(df))
    print("Categories:\n", df.category.value_counts().to_string())
    print("Risks:\n", df.risk.value_counts().to_string())
    for label in ["category","risk"]:
        counts = df[label].value_counts()
        if len(counts) < 2 or counts.min() < 5:
            raise ValueError(f"Not enough samples per class for {label}: {counts.to_dict()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "(CPU works but training may be slower)" if device.type=="cpu" else "")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for label in ["category","risk"]: train_one_label(df, label, tokenizer, device)

if __name__ == "__main__": main()
