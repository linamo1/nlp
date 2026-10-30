import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load and label
fake = pd.read_csv(r"C:\Users\HP\Downloads\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Downloads\True.csv")
fake["label"] = "FAKE"
true["label"] = "REAL"
df = pd.concat([fake, true], ignore_index=True)

# 2. Drop columns that leak the label
df = df.drop(columns=["subject", "date"])

# 3. Remove empty rows and duplicates
df = df.dropna(subset=["title", "text"])
df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"])

# 4. Strip source giveaways
def clean(text):
    text = re.sub(r"^.{0,60}\(Reuters\)\s*-\s*", "", text)  # "WASHINGTON (Reuters) - "
    text = re.sub(r"https?://\S+", "", text)                # URLs
    text = re.sub(r"<[^>]+>", "", text)                     # HTML tags
    text = re.sub(r"\b(Reuters|Getty Images|Featured image)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean)

# 5. Match your RSS input: title + first ~2 sentences
def short_version(row):
    first_sentences = " ".join(re.split(r"(?<=[.!?])\s+", row["text"])[:2])
    return f'{row["title"]}. {first_sentences}'

df["content"] = df.apply(short_version, axis=1)

# 6. Shuffle and split
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 7. Vectorize and train
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=5, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print(classification_report(y_test, model.predict(X_test_vec)))

# 8. Save in the format your main script loads
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f )
