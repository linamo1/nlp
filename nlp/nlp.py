import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# 1. Load and Label Data
# =====================================================

fake = pd.read_csv(r"C:\Users\HP\Downloads\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Downloads\True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], ignore_index=True)

# =====================================================
# 2. Drop Unnecessary Columns
# =====================================================

df = df.drop(columns=["subject", "date"])

# =====================================================
# 3. Remove Empty Rows and Duplicates
# =====================================================

df = df.dropna(subset=["title", "text"])
df = df[df["text"].str.strip() != ""]
df = df.drop_duplicates(subset=["text"])

# =====================================================
# 4. Text Cleaning
# =====================================================

def clean(text):
    text = re.sub(r"^.{0,60}\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(
        r"\b(Reuters|Getty Images|Featured image)\b",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean)

# =====================================================
# 5. Create Content Column
# =====================================================

def short_version(row):
    first_sentences = " ".join(
        re.split(r"(?<=[.!?])\s+", row["text"])[:2]
    )
    
    return f"{row['title']}. {first_sentences}"

df["content"] = df.apply(short_version, axis=1)

# =====================================================
# 6. Shuffle and Split Data
# =====================================================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =====================================================
# 7. TF-IDF Vectorization
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# Model 1: Random Forest (Bagging)
# =====================================================

print("\n==============================")
print("Random Forest Results")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_vec, y_train)

rf_predictions = rf_model.predict(X_test_vec)

print(classification_report(y_test, rf_predictions))
print("Accuracy:",
      accuracy_score(y_test, rf_predictions))

# =====================================================
# Model 2: AdaBoost (Boosting)
# =====================================================

print("\n==============================")
print("AdaBoost Results")
print("==============================")

base_model = DecisionTreeClassifier(max_depth=1)

ada_model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    random_state=42
)

ada_model.fit(X_train_vec.toarray(), y_train)

ada_predictions = ada_model.predict(
    X_test_vec.toarray()
)

print(classification_report(y_test, ada_predictions))
print("Accuracy:",
      accuracy_score(y_test, ada_predictions))

# =====================================================
# Save Best Model (AdaBoost)
# =====================================================

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(ada_model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully.")
