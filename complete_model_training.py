
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os

base_dir = r"D:\PYTHON\CropsDiseaseProject\leaf_and_non_leaf"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32
initial_epochs = 10
fine_tune_epochs = 10
fine_tune_at = 100

# DATA GENERATORS
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# BASE MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# Custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# COMPILE & TRAIN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\n Stage 1: Training base model for Leaf vs Non-Leaf classification")
history1 = model.fit(train_data, validation_data=val_data, epochs=initial_epochs)

# FINE-TUNING
print("\n Starting fine-tuning stage...")
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='binary_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_data, validation_data=val_data, epochs=fine_tune_epochs)

# EVALUATE
print("\n Evaluating on test data...")
test_loss, test_acc = model.evaluate(test_data)
print(f" Final Test Accuracy: {test_acc:.4f}")

# SAVE MODELS
model.save("Leaf_NonLeaf_Classifier.h5")
print(" Saved Keras model: Leaf_NonLeaf_Classifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("Leaf_NonLeaf_Classifier.tflite", 'wb') as f:
    f.write(tflite_model)
print(" Saved TFLite model: Leaf_NonLeaf_Classifier.tflite")

# PLOT RESULTS
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy', marker='o')
plt.plot(val_acc, label='Validation Accuracy', marker='o')
plt.title('Leaf vs Non-Leaf - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='o')
plt.title('Leaf vs Non-Leaf - Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
