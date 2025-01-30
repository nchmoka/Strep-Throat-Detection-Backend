import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score
)

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
BASE_LEARNING_RATE = 0.001
MODEL_DIR = 'models'

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Create data generators with augmentation during training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Create datasets
train_ds = train_datagen.flow_from_directory(
    'processed_data/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_ds = valid_test_datagen.flow_from_directory(
    'processed_data/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_ds = valid_test_datagen.flow_from_directory(
    'processed_data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Build the model with the same architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Create callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'strep_throat_cnn.keras'),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Calculate class weights
total = len(train_ds.classes)
pos = np.sum(train_ds.classes)
neg = total - pos
weight_for_0 = (1 / neg) * total / 2
weight_for_1 = (1 / pos) * total / 2
class_weights = {0: weight_for_0, 1: weight_for_1}

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Model Evaluation and Visualization Functions
def plot_training_history(history, save_dir):
    """Plot training history: loss and accuracy"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_pred_prob, save_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_prob, save_dir):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
    plt.close()

# Evaluate model and generate plots
print("\nGenerating evaluation metrics and plots...")

# Get predictions on test set
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_ds.classes

# Generate and save all plots
plot_training_history(history, MODEL_DIR)
plot_confusion_matrix(y_true, y_pred, MODEL_DIR)
plot_roc_curve(y_true, y_pred_prob, MODEL_DIR)
plot_precision_recall_curve(y_true, y_pred_prob, MODEL_DIR)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Calculate and print additional metrics
test_loss, test_accuracy = model.evaluate(test_ds)
roc_auc_score = auc(*roc_curve(y_true, y_pred_prob)[:2])
avg_precision = average_precision_score(y_true, y_pred_prob)

print("\nAdditional Metrics:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc_score:.4f}")
print(f"Average Precision Score: {avg_precision:.4f}")

# Save metrics to a file
with open(os.path.join(MODEL_DIR, 'model_metrics.txt'), 'w') as f:
    f.write("Model Evaluation Metrics\n")
    f.write("=======================\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"ROC AUC Score: {roc_auc_score:.4f}\n")
    f.write(f"Average Precision Score: {avg_precision:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, y_pred))

print("\nEvaluation completed! Check the 'models' directory for all plots and metrics.")