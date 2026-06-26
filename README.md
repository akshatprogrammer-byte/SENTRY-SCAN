# SENTRY/SCAN — Spam Message Threat Scanner

A Streamlit UI wrapped around your retrained TF-IDF + Logistic Regression spam
classifier (trained on `spam_ham_dataset.csv`, the Enron-style email spam/ham set).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `model.pkl` and `vectorizer.pkl` in the same folder as `app.py` — the app
loads them with `joblib.load()` on startup.

## Label mapping (important — this was the source of the earlier confusion)

This model's labels come straight from `spam_ham_dataset.csv`'s own `label_num`
column:

| label_num | label |
|---|---|
| 0 | ham  |
| 1 | spam |

That's the **opposite** convention from the first model you gave me (the SMS
Spam Collection one), which used `0 = spam, 1 = ham`. Two different datasets,
two different conventions — both correct for their own training data. This app
is wired to `SPAM_LABEL, HAM_LABEL = 1, 0` to match **this** model.

Verified directly against the trained model before shipping:
- Obvious promo/phishing text → predicts `1` (spam) ✓
- Obvious office email → predicts `0` (ham) ✓
- The bank-phishing example that the *old* SMS model missed → this model
  correctly flags it as spam (~72% confidence), since it's trained on
  email-style spam rather than SMS promo spam.

## What it does

- Paste or load a sample message, hit **RUN SCAN**.
- Vectorizes with your `TfidfVectorizer` (45,256-term vocabulary) and scores
  with your `LogisticRegression` model.
- Shows the verdict, a confidence meter, and the actual top contributing
  words pulled straight from the model's learned coefficients — for this
  model, **positive coefficient → spam, negative → ham** (confirmed
  empirically: words like `enron`, `attached`, `thanks` are strongly negative;
  `viagra`, `click`, `free`, `offer` are strongly positive).
- Annotated readback of the message with spam/ham signal words highlighted inline.

## Swapping in a different model later

If you retrain again, just drop the new `model.pkl` / `vectorizer.pkl` in.
But **re-check the label mapping every time** — as this round showed, it's
defined by however the labels were encoded in *that* training run, not by
convention. Quick check:

```python
import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
X = vectorizer.transform(["WINNER! free cash prize, click now"])
print(model.predict(X))  # whichever class this obvious-spam example
                          # predicts is your SPAM_LABEL
```

Then update `SPAM_LABEL, HAM_LABEL` near the top of `app.py` to match, and flip
the `COEF` sign comments/usages accordingly (search for `weight >` / `weight <`
in `app.py` — for binary `LogisticRegression`, positive coefficients always
push toward `classes_[1]`, so the sign convention flips whenever which class
is "spam" flips).
