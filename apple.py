import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import os
import random

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 15

# Paths to your dataset (organized as class-subfolders)
train_dir = 'D:/apple_disease_model/dataset_split/train'
val_dir = 'D:/apple_disease_model/dataset_split/val'
test_dir = 'D:/apple_disease_model/dataset_split/test'

# ============================================================
# DATA AUGMENTATION & GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ============================================================
# MODEL: TRANSFER LEARNING WITH MOBILENETV2
# ============================================================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# ============================================================
# TRAIN (FIRST PHASE: frozen base model)
# ============================================================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ============================================================
# FINE-TUNING (SECOND PHASE: unfreeze top layers of base model)
# ============================================================
base_model.trainable = True
for layer in base_model.layers[:-50]:  # keep lower layers frozen
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_generator,
    epochs=5,   # a few extra epochs
    validation_data=validation_generator,
    callbacks=callbacks
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model.save('apple_disease_mobilenet_final.h5')

# ============================================================
# TRAINING HISTORY PLOTS
# ============================================================
acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
loss = history.history['loss'] + fine_tune_history.history['loss']
val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# ============================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================
print("\n" + "="*60)
print("GENERATING CONFUSION MATRIX AND PERFORMANCE REPORT")
print("="*60)

class_names = list(train_generator.class_indices.keys())
print("Class names:", class_names)

print("Making predictions on test set...")
test_predictions = model.predict(test_generator)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = test_generator.classes

print(f"Total test samples: {len(test_true_classes)}")

cm = confusion_matrix(test_true_classes, test_pred_classes)
print("Confusion matrix generated!")

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names,
                     yticklabels=class_names,
                     cbar_kws={'label': 'Number of Images'},
                     annot_kws={'size': 12})

plt.title('Confusion Matrix - Apple Disease Classification\n', fontsize=16, fontweight='bold')
plt.ylabel('Actual Disease', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Disease', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)

readable_class_names = [name.replace('Apple___', '').replace('_', ' ') for name in class_names]

report = classification_report(test_true_classes, test_pred_classes, 
                               target_names=readable_class_names, digits=4)
print(report)

accuracy = np.sum(test_true_classes == test_pred_classes) / len(test_true_classes)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

print("\n" + "="*60)
print("PER-CLASS METRICS")
print("="*60)
for i, class_name in enumerate(readable_class_names):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp
    fn = np.sum(cm[i, :]) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{class_name:20} Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

# ============================================================
# SHOW SAMPLE PREDICTIONS WITH IMAGES
# ============================================================
sample_indices = random.sample(range(len(test_generator.filenames)), 8)  # show 8 random images
plt.figure(figsize=(16, 12))

for i, idx in enumerate(sample_indices):
    img_path = os.path.join(test_dir, test_generator.filenames[idx])
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100

    plt.subplot(2, 4, i+1)
    plt.imshow(load_img(img_path))
    plt.title(f"Pred: {readable_class_names[pred_class]}\nConf: {confidence:.2f}%", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=300)
plt.show()
