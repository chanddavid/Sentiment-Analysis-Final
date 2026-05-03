# model/train.py
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from model.data_cleaning import prepare_data

# 1. Load and clean data
df = prepare_data()
X = df["processed_review"].values
y = df["sentiment"].values

# 2. Train/validation/test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3. TextVectorization layer (modern replacement for Tokenizer + pad_sequences)
max_tokens = 10000
max_len = 100
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len,
    standardize='lower_and_strip_punctuation',
    split='whitespace'
)
vectorize_layer.adapt(X_train)

# 4. Build model using Functional API (includes vectorization layer)
inputs = Input(shape=(1,), dtype=tf.string)
x = vectorize_layer(inputs)
x = Embedding(max_tokens, 64)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 6. Save model and vectorizer's vocabulary (the layer is inside the model)
model.save('../sentiment_model.keras')
print("Model saved as ../sentiment_model.keras")

# 7. Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")