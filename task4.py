import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📌 Step 2: Load Dataset
# You can download the dataset from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 📌 Step 3: Preprocess Data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
# 📌 Step 4: Text Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 📌 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 📌 Step 6: Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 📌 Step 7: Predictions
y_pred = model.predict(X_test)

# 📌 Step 8: Evaluation
print("🔍 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# 📌 Step 9: Visualize Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
