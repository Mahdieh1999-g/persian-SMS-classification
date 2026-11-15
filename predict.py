
import os
import re
import sys
import unicodedata
import joblib
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

INPUT_FILE = os.path.join(ROOT, "sms_to_test.csv")
OUTPUT_FILE = os.path.join(ROOT, "sms_predictions.csv")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_LENGTH = 128


def normalize_persian(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("ي", "ی").replace("ى", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("\u200c", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_model(label):
    model_path = os.path.join(
        MODEL_DIR, f"{label}_model"
    )

    encoder_path = os.path.join(
        model_path, "label_encoder.joblib"
    )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"مدل {label} پیدا نشد. ابتدا train.py را اجرا کن."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    ).to(DEVICE)

    model.eval()

    encoder = joblib.load(encoder_path)

    return tokenizer, model, encoder


def predict_batch(texts, tokenizer, model, encoder):
    results = []

    # پیش‌بینی گروهی برای افزایش سرعت
    batch_size = 16

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        normalized = [
            normalize_persian(text)
            for text in batch_texts
        ]

        inputs = tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)

        for probs in probabilities:
            index = int(probs.argmax().item())

            label = encoder.inverse_transform([index])[0]
            score = float(probs[index].item())

            results.append((str(label), score))

    return results


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"فایل ورودی پیدا نشد:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    if "text" not in df.columns:
        raise ValueError(
            "فایل CSV باید ستونی با نام text داشته باشد."
        )

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()

    if df.empty:
        raise ValueError("هیچ پیامک معتبری در فایل وجود ندارد.")

    print(f"تعداد پیامک‌ها: {len(df)}")
    print(f"دستگاه اجرا: {DEVICE}")
    print("در حال بارگذاری مدل‌ها...")

    category_tokenizer, category_model, category_encoder = (
        load_model("category")
    )

    risk_tokenizer, risk_model, risk_encoder = (
        load_model("risk")
    )

    print("در حال دسته‌بندی پیامک‌ها...")

    category_results = predict_batch(
        df["text"].tolist(),
        category_tokenizer,
        category_model,
        category_encoder
    )

    risk_results = predict_batch(
        df["text"].tolist(),
        risk_tokenizer,
        risk_model,
        risk_encoder
    )

    df["predicted_category"] = [
        result[0] for result in category_results
    ]

    df["category_score"] = [
        round(result[1], 4) for result in category_results
    ]

    df["predicted_risk"] = [
        result[0] for result in risk_results
    ]

    df["risk_score"] = [
        round(result[1], 4) for result in risk_results
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nنتایج پیش‌بینی:")
    print(
        df[
            [
                "text",
                "predicted_category",
                "category_score",
                "predicted_risk",
                "risk_score"
            ]
        ].to_string(index=False)
    )

    print(f"\nفایل نتایج ذخیره شد:\n{OUTPUT_FILE}")
    print(
        "\nتوجه: امتیاز مدل لزوماً احتمال کالیبره‌شده نیست؛ "
        "پیش‌بینی به‌تنهایی اثبات کلاهبرداری یا امن بودن پیام نیست."
    )


if __name__ == "__main__":
    main()
