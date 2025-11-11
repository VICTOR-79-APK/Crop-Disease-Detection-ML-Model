import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tqdm import tqdm
import numpy as np
import shutil

# === CONFIGURATION ===
dataset_path = r"D:\PYTHON\CropsDiseaseProject\Datasets"
augment_count = 5
auto_clean = True  # 🔥 Set True to remove previously augmented images before re-running

# === IMAGE DATA GENERATOR ===
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

summary_report = {}

# === STEP 1: AUTO-CLEAN AUGMENTED IMAGES (if enabled) ===
if auto_clean:
    print("\n🧹 Cleaning previously augmented images...")
    deleted = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("_aug") or "_aug" in file:
                try:
                    os.remove(os.path.join(root, file))
                    deleted += 1
                except:
                    pass
    print(f"✅ Removed {deleted} old augmented images.\n")

# === STEP 2: CALCULATE DYNAMIC THRESHOLD ===
category_sizes = []
for crop in os.listdir(dataset_path):
    crop_path = os.path.join(dataset_path, crop)
    if not os.path.isdir(crop_path):
        continue

    for category in os.listdir(crop_path):
        category_path = os.path.join(crop_path, category)
        if os.path.isdir(category_path):
            count = len([f for f in os.listdir(category_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("_aug")])
            category_sizes.append(count)

if not category_sizes:
    print("⚠️ No valid images found in dataset!")
    exit()

dynamic_threshold = int(np.mean(category_sizes))
print(f"📏 Calculated Dynamic Threshold for Augmentation: {dynamic_threshold} images\n")

# === STEP 3: AUGMENTATION FUNCTION ===
def augment_category(category_path):
    images = [f for f in os.listdir(category_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("_aug")]
    img_count = len(images)

    if img_count >= dynamic_threshold:
        print(f"✅ {os.path.basename(category_path)} has {img_count} images — skipping augmentation.")
        return 0

    print(f"⚙️ Augmenting {os.path.basename(category_path)} ({img_count} images)...")
    augmented = 0

    for img_file in tqdm(images, desc=os.path.basename(category_path)):
        img_path = os.path.join(category_path, img_file)
        try:
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            prefix = os.path.splitext(img_file)[0]
            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=category_path,
                save_prefix=f"{prefix}_aug",
                save_format='jpg'
            ):
                i += 1
                augmented += 1
                if i >= augment_count:
                    break
        except Exception as e:
            print(f"⚠️ Skipped {img_file} due to error: {e}")
            continue
    return augmented


# === STEP 4: RUN AUGMENTATION FOR EACH CROP ===
for crop_folder in os.listdir(dataset_path):
    crop_path = os.path.join(dataset_path, crop_folder)
    if not os.path.isdir(crop_path):
        continue

    print(f"\n🌾 Processing Crop: {crop_folder}")
    summary_report[crop_folder] = {}

    for category in os.listdir(crop_path):
        category_path = os.path.join(crop_path, category)
        if not os.path.isdir(category_path):
            continue

        added = augment_category(category_path)
        summary_report[crop_folder][category] = added


# === STEP 5: FINAL SUMMARY REPORT ===
print("\n📊 Augmentation Summary Report 📋")
print("-" * 60)
total_added = 0
for crop, categories in summary_report.items():
    print(f"\n🌿 Crop: {crop}")
    for category, count in categories.items():
        print(f"   📂 {category:<25} → +{count} images added")
        total_added += count

print("-" * 60)
print(f"✅ Total New Augmented Images Added: {total_added}")
print(f"🎯 Augmentation Completed with Dynamic Threshold = {dynamic_threshold} images\n")
