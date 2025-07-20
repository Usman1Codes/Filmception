

import pandas as pd
import numpy as np
import ast # For safely evaluating string representations of lists
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel # TFDistilBertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW # AdamW is in optimizers.experimental in older TF
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, classification_report

import tensorflow as tf

# Check for GPU devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available!")
    print("GPU Details:", gpus[0])
    # Print GPU name (should include 'T4' or similar)
    print("GPU Name:", gpus[0].name)
else:
    print("No GPU found. Using CPU.")

"""## Configration"""

# --- Configuration ---
PRETRAINED_MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 256 # Max sequence length for tokenizer
BATCH_SIZE = 8 # Adjusted for potentially small dataset / memory
EPOCHS = 3 # Small number for quick demo; increase for real training
LEARNING_RATE = 3e-5
RANDOM_STATE = 42

"""## Data Loading"""

# --- 1. Data Loading ---
print("\n--- 1. Data Loading ---")
try:
    df = pd.read_csv("/content/cleaned_movie_data (1).csv")
except FileNotFoundError:
    print("Error: movie_genres_dataset.csv not found. Please create it first (e.g., by running the cell above).")
    exit()

print(f"Loaded dataset with {len(df)} samples.")
print("Sample data:")
print(df.head(2))

# The 'genres' column is a string representation of a list. Convert it.
# The 'labels' column is a string representation of a multi-hot vector.
# We will derive labels from 'genres' as per workflow.
df['genres_list'] = df['genres'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

"""### Label Processing"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# --- 2. Label Processing ---
print("\n--- 2. Label Processing ---")

# Clean genres: Remove spaces and apply specific replacements
cleaned_genres_list = df['genres_list'].apply(
    lambda genres: [
        g.replace(' ', '').replace('Action/Adventure', 'ActionAdventure').replace('Black-and-white', 'Blackandwhite')
        for g in genres
    ]
)

# Binarize the cleaned genres
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(cleaned_genres_list)
num_genres = len(mlb.classes_)
genre_names = mlb.classes_

# Print results
print(f"Number of unique genres: {num_genres}")
print(f"Genre names (mapping): {genre_names}")
print("Shape of Y (labels):", Y.shape)
print("Sample multi-hot encoded labels (first 2):")
print(Y[:2])

"""### # --- 3. Text Processing & Vectorization ---"""

# --- 3. Text Processing & Vectorization ---
print("\n--- 3. Text Processing & Vectorization ---")
tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

summaries = df['summary'].tolist()

# Tokenize
# We need to handle the case where summaries might not be strings (e.g. NaN)
# Ensure summaries are strings
summaries = [str(s) if pd.notnull(s) else "" for s in summaries]

X_tokenized = tokenizer(
    summaries,
    padding='max_length',
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors='tf' # TensorFlow tensors
)

input_ids = X_tokenized['input_ids']
attention_mask = X_tokenized['attention_mask']

print("Shape of input_ids:", input_ids.shape)
print("Shape of attention_mask:", attention_mask.shape)

"""## Data Splitting"""

#  --- 4. Data Splitting ---
print("\n--- 4. Data Splitting ---")
# Split ratio: 80% train, 10% validation, 10% test
TEST_VALID_SIZE = 0.2 # for the first split (gives 20% for temp)
TEST_SIZE_FROM_TEMP = 0.5 # for the second split (50% of 20% is 10% for final test)
RANDOM_STATE = 42

# First split: train vs. temp (temp will become validation + test)
X_ids_train, X_ids_temp, \
X_mask_train, X_mask_temp, \
y_train, y_temp = train_test_split(
    input_ids.numpy(), attention_mask.numpy(), Y, # Convert tensors to numpy for scikit-learn
    test_size=TEST_VALID_SIZE,
    random_state=RANDOM_STATE
)

# Second split: validation vs. test from temp
X_ids_val, X_ids_test, \
X_mask_val, X_mask_test, \
y_val, y_test = train_test_split(
    X_ids_temp, X_mask_temp, y_temp,
    test_size=TEST_SIZE_FROM_TEMP,
    random_state=RANDOM_STATE
)

print(f"Train set shapes: IDs {X_ids_train.shape}, Mask {X_mask_train.shape}, Labels {y_train.shape}")
print(f"Validation set shapes: IDs {X_ids_val.shape}, Mask {X_mask_val.shape}, Labels {y_val.shape}")
print(f"Test set shapes: IDs {X_ids_test.shape}, Mask {X_mask_test.shape}, Labels {y_test.shape}")

# Convert numpy arrays back to TensorFlow tensors for model input
X_train_tf = {'input_ids': tf.constant(X_ids_train), 'attention_mask': tf.constant(X_mask_train)}
X_val_tf = {'input_ids': tf.constant(X_ids_val), 'attention_mask': tf.constant(X_mask_val)}
X_test_tf = {'input_ids': tf.constant(X_ids_test), 'attention_mask': tf.constant(X_mask_test)}

y_train_tf = tf.constant(y_train, dtype=tf.float32) # Ensure float32 for loss calculation
y_val_tf = tf.constant(y_val, dtype=tf.float32)
y_test_tf = tf.constant(y_test, dtype=tf.float32)

"""## Model Selection"""



# --- 5. Model Selection & Architecture ---
print("\n--- 5. Model Selection & Architecture ---")

import tensorflow as tf
from transformers import TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda

# Load pre-trained DistilBERT model
base_model = TFDistilBertModel.from_pretrained(PRETRAINED_MODEL_NAME)

# Set whether to freeze the base model
freeze_base = False
if freeze_base:
    base_model.trainable = False  # Freeze base model layers
else:
    base_model.trainable = True  # Unfreeze for training

# Define input layers
input_ids_layer = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='input_ids')
attention_mask_layer = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name='attention_mask')

# Wrap DistilBERT call in a Lambda layer to handle KerasTensor inputs
def distilbert_call(inputs):
    input_ids, attention_mask = inputs
    return base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

cls_token_output = Lambda(distilbert_call, output_shape=(base_model.config.hidden_size,))([input_ids_layer, attention_mask_layer])

# Add custom classification head
x = Dropout(0.2)(cls_token_output)
output_layer = Dense(num_genres, activation='sigmoid', name='output_genres')(x)  # Sigmoid for multi-label

# Construct the model
model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output_layer)

# Print model summary
model.summary()

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import AdamW  # Requires `pip install keras`
import tensorflow.keras.backend as K

# Custom Multi-Label F1 Score Metric
class MultiLabelF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='micro', threshold=0.5, name='f1_score', **kwargs):
        super(MultiLabelF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        true_pos = tf.reduce_sum(y_true * y_pred, axis=0)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)
        self.false_negatives.assign_add(false_neg)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        if self.average == 'micro':
            total_tp = tf.reduce_sum(self.true_positives)
            total_fp = tf.reduce_sum(self.false_positives)
            total_fn = tf.reduce_sum(self.false_negatives)
            micro_precision = total_tp / (total_tp + total_fp + K.epsilon())
            micro_recall = total_tp / (total_tp + total_fn + K.epsilon())
            return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + K.epsilon())
        elif self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'weighted':
            weights = self.true_positives + self.false_negatives
            return tf.reduce_sum(f1 * weights) / (tf.reduce_sum(weights) + K.epsilon())
        else:
            raise ValueError("Unsupported average type. Use 'micro', 'macro', or 'weighted'.")

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))

# --- 6. Training ---
print("\n--- 6. Training ---")
LEARNING_RATE = 3e-5  # For AdamW, suitable for fine-tuning
EPOCHS = 5  # Increase for real dataset
BATCH_SIZE = 4  # Adjust based on GPU memory (small for small dataset)
num_genres = 33  # Define num_genres based on your dataset (number of genre labels)

# Loss Function
loss_function = BinaryCrossentropy()

# Optimizer
optimizer = AdamW(learning_rate=LEARNING_RATE)

# Metrics
metrics_list = [
    BinaryAccuracy(name='bin_accuracy'),
    MultiLabelF1Score(num_classes=num_genres, average='micro', threshold=0.5, name='f1_micro'),
    MultiLabelF1Score(num_classes=num_genres, average='macro', threshold=0.5, name='f1_macro'),
    MultiLabelF1Score(num_classes=num_genres, average='weighted', threshold=0.5, name='f1_weighted')
]

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_f1_micro',  # Monitor validation F1 micro
    patience=3,
    mode='max',  # F1 score should be maximized
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    filepath='best_genre_model.keras',  # Use .keras format
    save_best_only=True,
    monitor='val_f1_micro',  # Monitor validation F1 micro
    mode='max'
)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)

print("\nStarting training...")
# Note: With such a small dataset, training results might not be very meaningful.
# The validation set is tiny (1 sample if total is 10).
# For demonstration, we proceed.
history = model.fit(
    X_train_tf,
    y_train_tf,
    validation_data=(X_val_tf, y_val_tf),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint]
)

print("\nTraining finished.")
print(f"Best validation F1 Micro: {max(history.history['val_f1_micro'] if 'val_f1_micro' in history.history else [0]):.4f}")

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import AdamW
import tensorflow.keras.backend as K

# Custom Multi-Label F1 Score Metric
class MultiLabelF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, average='micro', threshold=0.5, name='f1_score', **kwargs):
        super(MultiLabelF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)

        true_pos = tf.reduce_sum(y_true * y_pred, axis=0)
        false_pos = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        false_neg = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)
        self.false_negatives.assign_add(false_neg)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

        if self.average == 'micro':
            total_tp = tf.reduce_sum(self.true_positives)
            total_fp = tf.reduce_sum(self.false_positives)
            total_fn = tf.reduce_sum(self.false_negatives)
            micro_precision = total_tp / (total_tp + total_fp + K.epsilon())
            micro_recall = total_tp / (total_tp + total_fn + K.epsilon())
            return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + K.epsilon())
        elif self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'weighted':
            weights = self.true_positives + self.false_negatives
            return tf.reduce_sum(f1 * weights) / (tf.reduce_sum(weights) + K.epsilon())
        else:
            raise ValueError("Unsupported average type. Use 'micro', 'macro', or 'weighted'.")

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))

# --- 6. Training ---
print("\n--- 6. Training ---")
LEARNING_RATE = 3e-5  # For AdamW, suitable for fine-tuning
EPOCHS = 10  # Increase for real dataset
BATCH_SIZE = 8  # Adjust based on GPU memory
num_genres = 33  # Define num_genres based on your dataset

# Loss Function
loss_function = BinaryCrossentropy()

# Optimizer
optimizer = AdamW(learning_rate=LEARNING_RATE)

# Metrics
metrics_list = [
    BinaryAccuracy(name='bin_accuracy'),
    MultiLabelF1Score(num_classes=num_genres, average='micro', threshold=0.5, name='f1_micro'),
    MultiLabelF1Score(num_classes=num_genres, average='macro', threshold=0.5, name='f1_macro'),
    MultiLabelF1Score(num_classes=num_genres, average='weighted', threshold=0.5, name='f1_weighted')
]

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_f1_micro',  # Monitor validation F1 micro
    patience=3,
    mode='max',  # F1 score should be maximized
    restore_best_weights=True
)

# Change the saved model name here
model_checkpoint = ModelCheckpoint(
    filepath='final_genre_model.keras',  # New model name
    save_best_only=True,
    monitor='val_f1_micro',  # Monitor validation F1 micro
    mode='max'
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=2,
    verbose=1,
    mode='min'
)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)

print("\nStarting training...")

# Fit the model
history = model.fit(
    X_train_tf,
    y_train_tf,
    validation_data=(X_val_tf, y_val_tf),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Training results
print("\nTraining finished.")
print(f"Best validation F1 Micro: {max(history.history['val_f1_micro'] if 'val_f1_micro' in history.history else [0]):.4f}")

# Save the final model manually after training
model.save('final_model.keras')  # Save model as 'final_model.keras'

# --- 7. Evaluation ---
print("\n--- 7. Evaluation ---")
# Load the best model saved by ModelCheckpoint
print("Loading best model for evaluation...")

from tensorflow.python.keras.utils.generic_utils import register_keras_serializable, _GLOBAL_CUSTOM_OBJECTS

if 'distilbert_call' not in _GLOBAL_CUSTOM_OBJECTS:
    # @register_keras_serializable()  # Only register if not already registered
    def distilbert_call(inputs):
        input_ids, attention_mask = inputs

        return base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

from tensorflow.keras.models import load_model

best_model = load_model(
    'final_model.keras',
    custom_objects={'distilbert_call': distilbert_call},
    compile=False   # <<< this skips metric & optimizer loading
)


# Generate predictions on the test set
y_pred_probs = best_model.predict(X_test_tf)

# Apply a threshold (typically 0.5) to get binary predictions
THRESHOLD = 0.5
y_pred_binary = (y_pred_probs > THRESHOLD).astype(int)

print("\nTest Set Evaluation Metrics:")
# Exact Match Ratio (subset accuracy)
emr = accuracy_score(y_test, y_pred_binary)
print(f"Exact Match Ratio (Accuracy): {emr:.4f}")

# Hamming Loss
hl = hamming_loss(y_test, y_pred_binary)
print(f"Hamming Loss: {hl:.4f}")

# F1 Scores
f1_micro = f1_score(y_test, y_pred_binary, average='micro', zero_division=0)
f1_macro = f1_score(y_test, y_pred_binary, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred_binary, average='weighted', zero_division=0)
print(f"F1 Score (Micro): {f1_micro:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Classification Report (per-class precision, recall, F1)
print("\nClassification Report (Test Set):")
# Ensure target_names are correctly ordered strings
report = classification_report(y_test, y_pred_binary, target_names=genre_names, zero_division=0)
print(report)

"""## Prediction"""

# --- 8. Prediction ---
print("\n--- 8. Prediction ---")

def predict_genres_for_summary(summary_text, tokenizer, model, mlb_instance, max_length=MAX_LENGTH, threshold=THRESHOLD):
    # 1. Preprocess (Tokenize)
    tokenized_input = tokenizer(
        [summary_text], # Needs to be a list
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )
    predict_payload = {'input_ids': tokenized_input['input_ids'],
                       'attention_mask': tokenized_input['attention_mask']}

    # 2. Predict probabilities
    probabilities = model.predict(predict_payload)

    # 3. Apply threshold
    binary_predictions = (probabilities > threshold).astype(int)

    # 4. Convert back to genre names
    # `mlb_instance.inverse_transform` expects a 2D array
    predicted_genre_list = mlb_instance.inverse_transform(binary_predictions)

    return predicted_genre_list[0] # Return the list of genres for the single summary


# Example new summary
new_summary_1 = "A brilliant scientist builds a time machine and travels to the future, only to find a dystopian society. He must fight to return to his own time and warn humanity.."
predicted_genres_1 = predict_genres_for_summary(new_summary_1, tokenizer, best_model, mlb)
print(f"\nSummary 1: '{new_summary_1[:100]}...'")
print(f"Predicted Genres 1: {predicted_genres_1}")

new_summary_2 = "Two young lovers from feuding families in Verona find their romance doomed by circumstances and their families' hate. A classic tale of love and tragedy."
predicted_genres_2 = predict_genres_for_summary(new_summary_2, tokenizer, best_model, mlb)
print(f"\nSummary 2: '{new_summary_2[:100]}...'")
print(f"Predicted Genres 2: {predicted_genres_2}")

new_summary_3 = "A funny dog goes on an adventure with his friends. They laugh a lot and eat snacks. Suitable for all ages."
predicted_genres_3 = predict_genres_for_summary(new_summary_3, tokenizer, best_model, mlb)
print(f"\nSummary 3: '{new_summary_3[:100]}...'")
print(f"Predicted Genres 3: {predicted_genres_3}")

print("\n--- Workflow Implementation Complete ---")