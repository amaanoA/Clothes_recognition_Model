vgg16
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

print("="*60)
print("VGG16 TRAINING - PAPER METHODOLOGY")
print("="*60)

# Configuration 
EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
IMG_SIZE = (224, 224)

# Dataset Paths
BASE_DIR = r'D:\clothes regognition fine tuned\data'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VAL_DIR = os.path.join(BASE_DIR, 'Val')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

# Save Directory
os.makedirs('saved_models/vgg16', exist_ok=True)

# Data Augmentation (Real-time) - Matching Paper
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
print(f"\nDetected {num_classes} classes: {class_names}")

# Save class names
with open('saved_models/vgg16/class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Build VGG16 Model
print("\nBuilding VGG16 model...")
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Add classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTotal parameters: {model.count_params():,}")

# Callbacks
checkpoint = ModelCheckpoint(
    'saved_models/vgg16/best_vgg16.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Training
print("\nStarting VGG16 training...")
print("="*60)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

# Save final model
model.save('saved_models/vgg16/vgg16_final.h5')
print("\nTraining complete. Model saved.")

# Plot Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('VGG16 - Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('VGG16 - Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('saved_models/vgg16/vgg16_training_plots.jpg', dpi=300)
plt.close()
print("Training plots saved.")

# Evaluate on Test Set
print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

# Generate predictions
print("\nGenerating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*60)
print("VGG16 - EVALUATION METRICS")
print("="*60)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print("="*60)

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - VGG16', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('saved_models/vgg16/vgg16_confusion_matrix.jpg', dpi=300)
plt.close()
print("Confusion matrix saved.")

# Save metrics
with open('saved_models/vgg16/vgg16_metrics.txt', 'w') as f:
    f.write("VGG16 Training Results\n")
    f.write("="*60 + "\n\n")
    f.write("Configuration:\n")
    f.write(f"  - Architecture: VGG16\n")
    f.write(f"  - Epochs: {EPOCHS}\n")
    f.write(f"  - Batch Size: {BATCH_SIZE}\n")
    f.write(f"  - Learning Rate: {LEARNING_RATE}\n")
    f.write(f"  - Image Size: {IMG_SIZE}\n")
    f.write(f"  - Number of Classes: {num_classes}\n")
    f.write(f"  - Total Parameters: {model.count_params():,}\n\n")
    f.write("Evaluation Metrics:\n")
    f.write(f"  - Test Accuracy:  {accuracy*100:.2f}%\n")
    f.write(f"  - Test Precision: {precision*100:.2f}%\n")
    f.write(f"  - Test Recall:    {recall*100:.2f}%\n")
    f.write(f"  - Test F1-Score:  {f1*100:.2f}%\n")
    f.write(f"  - Test Loss:      {test_loss:.4f}\n\n")
    f.write("="*60 + "\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

print("\nAll results saved to 'saved_models/vgg16/' directory.")
print("VGG16 training complete!")