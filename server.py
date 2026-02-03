import os
import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from google.cloud import vision

#  Initialize FastAPI App

app = FastAPI(title="Crop Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  MODEL LOADING

MODEL_DIR = r"D:\PYTHON\CropsDiseaseProject\models"

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

models = {}
for crop, model_path in CROP_MODELS.items():
    if os.path.exists(model_path):
        models[crop] = tf.keras.models.load_model(model_path)
        print(f" Loaded model for {crop}")
    else:
        print(f" Model not found for {crop}: {model_path}")


#  Google Vision Leaf Detection

def is_leaf_image_google(img: Image.Image, confidence_threshold=0.6):
    """
    Uses Google Cloud Vision API to check if image contains a leaf or plant.
    Returns True if likely a leaf/plant, otherwise False.
    """
    try:
        # Convert PIL image to bytes
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        content = buf.getvalue()

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)

        response = client.label_detection(image=image)
        labels = response.label_annotations

        print("\n Google Vision detected labels:")
        for label in labels:
            print(f"{label.description:25s} | Confidence: {label.score:.2f}")

        # Check for keywords that indicate a leaf or plant
        leaf_keywords = ["leaf", "plant", "foliage", "tree", "botany", "vegetation", "crop"]
        for label in labels:
            if any(keyword in label.description.lower() for keyword in leaf_keywords) and label.score >= confidence_threshold:
                print(f" Leaf-like image detected ({label.description}, {label.score:.2f})")
                return True

        print(" No strong leaf/plant-related labels found.")
        return False

    except Exception as e:
        print(f" Vision API error: {e}")
        return True
    
#  Disease Prediction Function

def predict_crop_disease(crop: str, image: Image.Image):
    if crop not in models:
        return {"error": f"No model found for crop: {crop}"}

    model = models[crop]
    img = image.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    predicted_index = np.argmax(preds[0])

    test_dir = os.path.join(r"D:\PYTHON\CropsDiseaseProject\Data_Split\test", crop)
    class_labels = sorted(os.listdir(test_dir))
    predicted_label = class_labels[predicted_index]
    disease_name = predicted_label.replace(f"{crop}_", "")

    return {
        "crop": crop,
        "disease": disease_name,
        "confidence": f"{np.max(preds[0]) * 100:.2f}%",
        "solution": "Apply appropriate pesticide or fungicide as per agricultural guidelines."
    }


# predict Endpoint — with Google Vision validation

@app.post("/predict")
async def predict(crop: str = Form(...), image: UploadFile = File(...)):
    """
    Main prediction endpoint:
    1️ Validate leaf image via Google Vision API
    2️ Predict crop disease using trained model
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Step 1 — Leaf validation
        if not is_leaf_image_google(img):
            return {
                "status": "error",
                "message": "Invalid image — please upload a clear leaf image."
            }

        # Step 2 — Predict disease
        result = predict_crop_disease(crop, img)
        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# debug_leaf_labels Endpoint — test Vision API labels

@app.post("/debug_leaf_labels")
async def debug_leaf_labels(image: UploadFile = File(...)):
    """
    Upload any image to view Vision API labels + confidence scores.
    Useful for debugging and tuning threshold.
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        content = buf.getvalue()

        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=content)
        
        response = client.label_detection(image=image)
        labels = response.label_annotations

        debug_data = [
            {"label": label.description, "confidence": round(label.score, 2)}
            for label in labels
        ]

        return {"status": "success", "labels": debug_data}

    except Exception as e:
        return {"status": "error", "message": str(e)}


