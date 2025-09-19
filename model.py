import pandas as pd
import re
import nltk
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ----------- Load Dataset ------------
data = pd.read_csv("News.csv", index_col=0)

# ----------- Preprocessing ------------
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text_data):
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', str(sentence))
        preprocessed_text.append(
            ' '.join(token.lower() for token in sentence.split() if token.lower() not in stop_words)
        )
    return preprocessed_text

# Drop unnecessary columns
data = data.drop(["title", "subject", "date"], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
data['text'] = preprocess_text(data['text'].values)

# ----------- Train/Test Split ------------
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['class'], test_size=0.25, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ----------- Logistic Regression Model ------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

print("Train Accuracy:", accuracy_score(y_train, log_model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, log_model.predict(X_test)))

# ----------- Save Model + Vectorizer ------------
with open("FakeNewsModel.pkl", "wb") as f:
    pickle.dump(log_model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and Vectorizer saved successfully!")
