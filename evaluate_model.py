import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# ========================
# ⚙️ CONFIGURATION
# ========================
BASE_DIR = r"D:\\PYTHON\\CropsDiseaseProject\\Data_Split\\test"
MODEL_DIR = r"D:\PYTHON\CropsDiseaseProject\models"
RESULTS_DIR = r"D:\PYTHON\CropsDiseaseProject\results"  # new folder to store results
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

CROP_MODELS = {
    "Chili": os.path.join(MODEL_DIR, "Chili_Model_FT.h5"),
    "Maize": os.path.join(MODEL_DIR, "Maize_Model_FT.h5"),
    "Pea": os.path.join(MODEL_DIR, "Pea_Model_FT.h5"),
    "Potato": os.path.join(MODEL_DIR, "Potato_Model_FT.h5"),
    "Rice": os.path.join(MODEL_DIR, "Rice_Model_FT.h5"),
    "Sugarcane": os.path.join(MODEL_DIR, "Sugarcane_Model_FT.h5"),
    "Tomato": os.path.join(MODEL_DIR, "Tomato_Model_FT.h5"),
    "Wheat": os.path.join(MODEL_DIR, "Wheat_Model_FT.h5"),
}

# ========================
# 🧠 DATA PREPROCESSOR
# ========================
datagen = ImageDataGenerator(rescale=1.0 / 255)
all_y_true, all_y_pred, all_labels = [], [], []

for crop_name, model_path in CROP_MODELS.items():
    print(f"\n🌾 Evaluating model for {crop_name}...")

    test_path = os.path.join(BASE_DIR, crop_name)
    if not os.path.exists(test_path):
        print(f"❌ Test path not found for {crop_name}: {test_path}")
        continue

    model = load_model(model_path)
    print(f"✅ Loaded model: {model_path}")

    test_gen = datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    # Avoid repeating crop name if already present
    crop_labels = [
        cls if cls.lower().startswith(crop_name.lower())
        else f"{crop_name}_{cls}" for cls in labels
    ]

    all_y_true.extend([crop_labels[y] for y in y_true])
    all_y_pred.extend([crop_labels[y] for y in y_pred])
    all_labels.extend(crop_labels)

# ========================
# 📊 METRICS
# ========================
print("\n✅ Generating Combined Evaluation Metrics...")
all_labels = sorted(list(set(all_labels)))

cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
f1 = f1_score(all_y_true, all_y_pred, average='weighted')
report = classification_report(all_y_true, all_y_pred)

# Print metrics
print(f"\n Combined Weighted F1 Score: {f1:.4f}")
print("\n📋 Classification Report:")
print(report)

# ========================
# 💾 SAVE RESULTS
# ========================
report_path = os.path.join(RESULTS_DIR, "combined_classification_report.txt")
cm_image_path = os.path.join(RESULTS_DIR, "combined_confusion_matrix.png")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
------

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Combined Weighted F1 Score: {:.4f}\n\n".format(f1))
    f.write(report)


print(f"\n📄 Report saved at: {report_path}")

# ========================
# 🔥 CONFUSION MATRIX PLOT
# ========================
num_labels = len(all_labels)
plt.figure(figsize=(max(14, num_labels // 1.3), max(12, num_labels // 1.3)))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.title("Combined Confusion Matrix — All Crops", fontsize=16)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(cm_image_path, dpi=300)
print(f"🖼️ Confusion matrix saved at: {cm_image_path}")

plt.show()
