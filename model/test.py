# model/test.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from model.data_cleaning import prepare_data

# 1. Load data
df = prepare_data()
X = df["processed_review"].values
y = df["sentiment"].values

# 2. Recreate test split (same as training)
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 3. Load model (contains vectorization layer)
model = load_model('../sentiment_model.keras')

# 4. Predict
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

# 5. Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))