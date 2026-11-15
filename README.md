# Persian SMS Classifier — Starter

This is a small starter project for experimenting with Persian SMS topic and risk classification offline.

## Important limitations
- The included dataset is synthetic and tiny. It is for testing the pipeline only, not for reliable real-world detection.
- The model predicts two separate labels:
  - `category`: banking, verification, shopping, bills, advertising, personal, suspicious, other
  - `risk`: normal, suspicious, high
- A message being categorized as `normal` does not prove it is safe. A message containing a link is not automatically a scam.
- Do not train on private SMS without informed consent and removing personal identifiers, OTPs, phone numbers, account numbers, and other sensitive data.

## Windows setup
Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -> or -> python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
python train.py
```

If PowerShell blocks activation, run this instead:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

## Files
- `data/sms_examples.csv`: tiny synthetic examples for pipeline testing
- `train.py`: normalizes Persian text and trains two TF-IDF + LinearSVC classifiers
- `predict.py`: loads the saved models and predicts category/risk for a typed message
- `models/`: created after training

## Next steps
Replace/expand the synthetic examples with a carefully reviewed, diverse dataset. Keep a separate test set that is never used for training. For scam detection, measure precision and recall for the suspicious/high-risk classes, not just overall accuracy.
