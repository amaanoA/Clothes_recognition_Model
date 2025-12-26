mobile v2 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

print("="*70)
print("MOBILENETV2 FINE-TUNING - STAGE 2 (Unfreezing Layers)")
print("="*70)
print("\nThis script will:")
print("1. Load your existing MobileNetV2 model")
print("2. Unfreeze layers for fine-tuning")
print("3. Train with very low learning rate")
print("4. Target: 85-92% accuracy")
print("="*70)

# Configuration
FINE_TUNE_EPOCHS = 35
BATCH_SIZE = 32
FINE_TUNE_LR = 0.00001
IMG_SIZE = (224, 224)

# Dataset Paths
BASE_DIR = r'D:\clothes regognition fine tuned\data'
TRAIN_DIR = os.path.join(BASE_DIR, 'Train')
VAL_DIR = os.path.join(BASE_DIR, 'Val')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

# Model Path
MODEL_PATH = 'saved_models/mobilenetv2/best_mobilenetv2.h5'

if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model not found at {MODEL_PATH}")
    print("Please train the initial model first using train_mobilenetv2.py")
    exit()

# Save Directory
os.makedirs('saved_models/mobilenetv2/finetuned', exist_ok=True)

# Data Augmentation
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
print("\nLoading datasets...")
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
print(f"Detected {num_classes} classes: {class_names}")

# Save class names
with open('saved_models/mobilenetv2/finetuned/class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Check class distribution
print("\nClass Distribution:")
class_counts = {}
for class_name, idx in train_generator.class_indices.items():
    count = list(train_generator.classes).count(idx)
    class_counts[class_name] = count
    print(f"  {class_name:20s}: {count:4d} samples")

min_count = min(class_counts.values())
max_count = max(class_counts.values())
if max_count / min_count > 3:
    print(f"\nWARNING: Imbalanced dataset detected!")
    print(f"Ratio: {max_count}/{min_count} = {max_count/min_count:.1f}x")

# Load Existing Model
print(f"\nLoading pre-trained model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

# Evaluate current performance
print("\nCurrent Model Performance (Before Fine-Tuning):")
initial_loss, initial_acc = model.evaluate(val_generator, verbose=0)
print(f"   Validation Accuracy: {initial_acc*100:.2f}%")
print(f"   Validation Loss: {initial_loss:.4f}")

# Unfreeze layers for fine-tuning
print("\nUnfreezing layers for fine-tuning...")

# Find MobileNetV2 base model
base_model = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    print("   Unfreezing ALL layers...")
    for layer in model.layers:
        layer.trainable = True
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
else:
    print("   Unfreezing last 50 layers of MobileNetV2...")
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])

print(f"   {trainable_count} layers are now trainable")

# Recompile with very low learning rate
print(f"\nRecompiling model with LR={FINE_TUNE_LR}")
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nModel Summary:")
print(f"   Total parameters: {model.count_params():,}")
print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Callbacks
checkpoint = ModelCheckpoint(
    'saved_models/mobilenetv2/finetuned/best_mobilenetv2_finetuned.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# Fine-Tuning
print("\n" + "="*70)
print("STARTING FINE-TUNING (STAGE 2)")
print("="*70)
print(f"Training for {FINE_TUNE_EPOCHS} epochs with LR={FINE_TUNE_LR}")
print("="*70 + "\n")

history = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

# Save final model
model.save('saved_models/mobilenetv2/finetuned/mobilenetv2_finetuned_final.h5')
print("\nFine-tuned model saved.")

# Plot Fine-Tuning History
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.axhline(y=initial_acc, color='gray', linestyle='--', label=f'Initial Val Acc ({initial_acc*100:.1f}%)')
plt.title('MobileNetV2 Fine-Tuning: Accuracy Improvement', fontsize=14, fontweight='bold')
plt.xlabel('Fine-Tuning Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.axhline(y=initial_loss, color='gray', linestyle='--', label=f'Initial Val Loss ({initial_loss:.2f})')
plt.title('MobileNetV2 Fine-Tuning: Loss Reduction', fontsize=14, fontweight='bold')
plt.xlabel('Fine-Tuning Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('saved_models/mobilenetv2/finetuned/finetuning_plots.jpg', dpi=300)
plt.close()
print("Fine-tuning plots saved.")

# Evaluate on Test Set
print("\n" + "="*70)
print("EVALUATING FINE-TUNED MODEL ON TEST SET")
print("="*70)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print("\n" + "="*70)
print("FINE-TUNING RESULTS SUMMARY")
print("="*70)
print(f"BEFORE Fine-Tuning: {initial_acc*100:.2f}%")
print(f"AFTER Fine-Tuning:  {test_accuracy*100:.2f}%")
print(f"Improvement:        +{(test_accuracy-initial_acc)*100:.2f}%")
print("="*70)

# Generate predictions
print("\nGenerating predictions for detailed metrics...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*70)
print("DETAILED EVALUATION METRICS")
print("="*70)
print(f"Test Accuracy:  {accuracy*100:.2f}%")
print(f"Test Precision: {precision*100:.2f}%")
print(f"Test Recall:    {recall*100:.2f}%")
print(f"Test F1-Score:  {f1*100:.2f}%")
print(f"Test Loss:      {test_loss:.4f}")
print("="*70)

# Check if target reached
if accuracy >= 0.85:
    print("\nSUCCESS! Achieved target accuracy (>=85%)")
elif accuracy >= 0.80:
    print("\nGood progress! Close to target (>=85%)")
    print("Suggestion: Try training for 10-20 more epochs")
else:
    print("\nBelow target. Possible improvements:")
    print("   1. Train for more epochs")
    print("   2. Check dataset quality and balance")
    print("   3. Unfreeze all layers")

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Per-Class Accuracy
print("\nPer-Class Performance:")
cm = confusion_matrix(y_true, y_pred)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
    status = "[OK]" if acc >= 0.80 else "[LOW]" if acc < 0.60 else "[MED]"
    print(f"   {status} {class_name:20s}: {acc*100:5.1f}%")

# Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - MobileNetV2 (Fine-Tuned)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('saved_models/mobilenetv2/finetuned/confusion_matrix_finetuned.jpg', dpi=300)
plt.close()
print("\nConfusion matrix saved.")

# Save comprehensive metrics report
with open('saved_models/mobilenetv2/finetuned/finetuning_report.txt', 'w') as f:
    f.write("MobileNetV2 Fine-Tuning Report\n")
    f.write("="*70 + "\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Initial Model: {MODEL_PATH}\n")
    f.write(f"  Fine-Tuning Epochs: {FINE_TUNE_EPOCHS}\n")
    f.write(f"  Learning Rate: {FINE_TUNE_LR}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Trainable Parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Before Fine-Tuning: {initial_acc*100:.2f}%\n")
    f.write(f"  After Fine-Tuning:  {accuracy*100:.2f}%\n")
    f.write(f"  Improvement:        +{(accuracy-initial_acc)*100:.2f}%\n\n")
    
    f.write("DETAILED METRICS:\n")
    f.write(f"  Test Accuracy:  {accuracy*100:.2f}%\n")
    f.write(f"  Test Precision: {precision*100:.2f}%\n")
    f.write(f"  Test Recall:    {recall*100:.2f}%\n")
    f.write(f"  Test F1-Score:  {f1*100:.2f}%\n")
    f.write(f"  Test Loss:      {test_loss:.4f}\n\n")
    
    f.write("PER-CLASS ACCURACY:\n")
    for class_name, acc in zip(class_names, per_class_acc):
        f.write(f"  {class_name:20s}: {acc*100:5.1f}%\n")
    
    f.write("\n" + "="*70 + "\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

print("\n" + "="*70)
print("FINE-TUNING COMPLETE!")
print("="*70)
print("\nSaved files:")
print("  - saved_models/mobilenetv2/finetuned/best_mobilenetv2_finetuned.h5")
print("  - saved_models/mobilenetv2/finetuned/finetuning_plots.jpg")
print("  - saved_models/mobilenetv2/finetuned/confusion_matrix_finetuned.jpg")
print("  - saved_models/mobilenetv2/finetuned/finetuning_report.txt")

if accuracy >= 0.91:
    print("\nEXCELLENT! Target accuracy (91%+) achieved!")
elif accuracy >= 0.85:
    print("\nGOOD! Close to paper's target (91%)")
else:
    print("\nNext Steps:")
    print("  1. Review per-class performance above")
    print("  2. Consider running additional fine-tuning epochs")
    print("  3. You can re-run this script to continue fine-tuning")

print("\n" + "="*70)