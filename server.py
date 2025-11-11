# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image
# import os

# # ======================================================
# # 🚀 Initialize FastAPI App
# # ======================================================
# app = FastAPI(title="Crop Disease Prediction API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ======================================================
# # ⚙ MODEL LOADING
# # ======================================================
# MODEL_DIR = r"D:\PYTHON\CropsDiseaseProject\models"

# # --- Load disease models for each crop ---
# CROP_MODELS = {
#     "Chili": os.path.join(MODEL_DIR, "Chili_Model_FT.h5"),
#     "Maize": os.path.join(MODEL_DIR, "Maize_Model_FT.h5"),
#     "Pea": os.path.join(MODEL_DIR, "Pea_Model_FT.h5"),
#     "Potato": os.path.join(MODEL_DIR, "Potato_Model_FT.h5"),
#     "Rice": os.path.join(MODEL_DIR, "Rice_Model_FT.h5"),
#     "Sugarcane": os.path.join(MODEL_DIR, "Sugarcane_Model_FT.h5"),
#     "Tomato": os.path.join(MODEL_DIR, "Tomato_Model_FT.h5"),
#     "Wheat": os.path.join(MODEL_DIR, "Wheat_Model_FT.h5"),
# }

# models = {}
# for crop, model_path in CROP_MODELS.items():
#     if os.path.exists(model_path):
#         models[crop] = tf.keras.models.load_model(model_path)
#         print(f" Loaded model for {crop}")
#     else:
#         print(f" Model not found for {crop}: {model_path}")

# # --- Load crop-type classifier model ---
# CROP_TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "crop_type_classifier.h5")
# crop_type_model = None
# if os.path.exists(CROP_TYPE_MODEL_PATH):
#     crop_type_model = tf.keras.models.load_model(CROP_TYPE_MODEL_PATH)
#     print(" Loaded Crop-Type Classifier model.")
# else:
#     print(" Crop-Type Classifier model not found!")

# # Crop class order (must match training order)
# CROP_CLASSES = ['Wheat', 'Rice', 'Maize', 'Potato', 'Tomato', 'Chili', 'Sugarcane', 'Pea']

# # ======================================================
# # 🧠 Helper — Crop-Type Verification
# # ======================================================
# def verify_crop_type(img: Image.Image, claimed_crop: str, threshold=0.55):
#     """Check if uploaded image belongs to the claimed crop using crop_type_classifier."""
#     if crop_type_model is None:
#         return True  # skip verification if model not loaded

#     img = img.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     preds = crop_type_model.predict(img_array)
#     predicted_index = np.argmax(preds[0])
#     predicted_crop = CROP_CLASSES[predicted_index]
#     confidence = np.max(preds[0])

#     print(f"Predicted Crop: {predicted_crop} | Confidence: {confidence:.2f}")

#     if predicted_crop.lower() != claimed_crop.lower() or confidence < threshold:
#         print(f" Invalid image — expected {claimed_crop}, got {predicted_crop}")
#         return False
#     print(f" Verified crop: {predicted_crop} ({confidence:.2f})")
#     return True


# # ======================================================
# # 🧩 Disease Prediction Function
# # ======================================================
# def predict_crop_disease(crop: str, image: Image.Image):
#     if crop not in models:
#         return {"error": f"No model found for crop: {crop}"}

#     model = models[crop]
#     img = image.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     preds = model.predict(img_array)
#     predicted_index = np.argmax(preds[0])

#     test_dir = os.path.join(r"D:\PYTHON\CropsDiseaseProject\Data_Split\test", crop)
#     class_labels = sorted(os.listdir(test_dir))
#     predicted_label = class_labels[predicted_index]
#     disease_name = predicted_label.replace(f"{crop}_", "")

#     return {
#         "crop": crop,
#         "disease": disease_name,
#         "confidence": f"{np.max(preds[0]) * 100:.2f}%",
#         "solution": "Apply proper pesticide/fungicide as per agricultural guidelines."
#     }


# # ======================================================
# # 🧪 /test Endpoint
# # ======================================================
# @app.post("/test")
# async def test_endpoint(crop: str = Form(...), image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")

#         if not verify_crop_type(img, crop):
#             return {"status": "error", "message": "Invalid image — does not belong to selected crop."}

#         result = predict_crop_disease(crop, img)
#         return {"status": "success", "data": result}

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# # ======================================================
# # /predict Endpoint (Main) — Enhanced Version
# # ======================================================
# @app.post("/predict")
# async def predict(crop: str = Form(...), image: UploadFile = File(...)):
#     """
#     Main endpoint — verifies crop type using classifier,
#     then predicts disease using the crop-specific model.
#     """
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")

#         # Step 1️⃣ — Verify crop type using classifier
#         is_valid = verify_crop_type(img, crop, threshold=0.55)
#         if not is_valid:
#             return {
#                 "status": "error",
#                 "message": f"Invalid image — the uploaded image does not belong to {crop} crop."
#             }

#         # Step 2️⃣ — Predict disease if valid crop
#         result = predict_crop_disease(crop, img)

#         # Step 3️⃣ — Return clean structured JSON
#         return {
#             "status": "success",
#             "data": {
#                 "crop": result["crop"],
#                 "disease": result["disease"],
#                 "confidence": result["confidence"],
#                 "solution": result["solution"]
#             }
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Server error: {str(e)}"
#         }



# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image
# import os

# # ======================================================
# # 🚀 Initialize FastAPI App
# # ======================================================
# app = FastAPI(title="Crop Disease Prediction API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ======================================================
# # ⚙ MODEL LOADING
# # ======================================================
# MODEL_DIR = r"D:\PYTHON\CropsDiseaseProject\models"

# # --- Load disease models for each crop ---
# CROP_MODELS = {
#     "Chili": os.path.join(MODEL_DIR, "Chili_Model_FT.h5"),
#     "Maize": os.path.join(MODEL_DIR, "Maize_Model_FT.h5"),
#     "Pea": os.path.join(MODEL_DIR, "Pea_Model_FT.h5"),
#     "Potato": os.path.join(MODEL_DIR, "Potato_Model_FT.h5"),
#     "Rice": os.path.join(MODEL_DIR, "Rice_Model_FT.h5"),
#     "Sugarcane": os.path.join(MODEL_DIR, "Sugarcane_Model_FT.h5"),
#     "Tomato": os.path.join(MODEL_DIR, "Tomato_Model_FT.h5"),
#     "Wheat": os.path.join(MODEL_DIR, "Wheat_Model_FT.h5"),
# }

# models = {}
# for crop, model_path in CROP_MODELS.items():
#     if os.path.exists(model_path):
#         models[crop] = tf.keras.models.load_model(model_path)
#         print(f"✅ Loaded model for {crop}")
#     else:
#         print(f"⚠️ Model not found for {crop}: {model_path}")

# # --- Load crop-type classifier model ---
# CROP_TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "crop_type_classifier.h5")
# crop_type_model = None
# if os.path.exists(CROP_TYPE_MODEL_PATH):
#     crop_type_model = tf.keras.models.load_model(CROP_TYPE_MODEL_PATH)
#     print("✅ Loaded Crop-Type Classifier model.")
# else:
#     print("⚠️ Crop-Type Classifier model not found!")

# # --- Load Leaf vs Non-Leaf model ---
# LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "Leaf_NonLeaf_Classifier.h5")
# leaf_classifier = None
# if os.path.exists(LEAF_MODEL_PATH):
#     leaf_classifier = tf.keras.models.load_model(LEAF_MODEL_PATH)
#     print("✅ Loaded Leaf vs Non-Leaf Classifier model.")
# else:
#     print("⚠️ Leaf vs Non-Leaf Classifier model not found!")

# # Crop class order (must match training order)
# CROP_CLASSES = ['Wheat', 'Rice', 'Maize', 'Potato', 'Tomato', 'Chili', 'Sugarcane', 'Pea']

# # ======================================================
# # 🧠 Helper — Leaf vs Non-Leaf Verification (Fixed Logic)
# # ======================================================
# def verify_leaf_image(img: Image.Image, threshold: float = 0.55):
#     """
#     Checks whether the uploaded image is a leaf or non-leaf.

#     threshold → confidence threshold for p_leaf
#     invert_output → set True if model output = p(nonleaf)
#                     set False if model output = p(leaf)
#     """

#     if leaf_classifier is None:
#         print("⚠️ Leaf vs Non-Leaf model not loaded, skipping leaf check.")
#         return True

#     img = img.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     pred = float(leaf_classifier.predict(img_array)[0][0])

#     # ✅ Change this after testing /debug_leaf_check
#     invert_output = True  # True → model outputs p(nonleaf); False → p(leaf)

#     if invert_output:
#         p_leaf = 1 - pred
#         p_nonleaf = pred
#     else:
#         p_leaf = pred
#         p_nonleaf = 1 - pred

#     print(f"🧪 Leaf Detector → p_leaf={p_leaf:.4f}, p_nonleaf={p_nonleaf:.4f}")

#     if p_leaf >= threshold:
#         print("✅ Detected as LEAF image.")
#         return True
#     else:
#         print("❌ Detected as NON-LEAF image.")
#         return False

# # ======================================================
# # 🧠 Helper — Crop-Type Verification
# # ======================================================
# def verify_crop_type(img: Image.Image, claimed_crop: str, threshold=0.55):
#     """Check if uploaded image belongs to the claimed crop using crop_type_classifier."""
#     if crop_type_model is None:
#         return True  # skip verification if model not loaded

#     img = img.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     preds = crop_type_model.predict(img_array)
#     predicted_index = np.argmax(preds[0])
#     predicted_crop = CROP_CLASSES[predicted_index]
#     confidence = np.max(preds[0])

#     print(f"🌾 Predicted Crop: {predicted_crop} | Confidence: {confidence:.2f}")

#     if predicted_crop.lower() != claimed_crop.lower() or confidence < threshold:
#         print(f"⚠️ Invalid crop image — expected {claimed_crop}, got {predicted_crop}")
#         return False

#     print(f"✅ Verified crop: {predicted_crop} ({confidence:.2f})")
#     return True


# # ======================================================
# # 🧩 Disease Prediction Function
# # ======================================================
# def predict_crop_disease(crop: str, image: Image.Image):
#     if crop not in models:
#         return {"error": f"No model found for crop: {crop}"}

#     model = models[crop]
#     img = image.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     preds = model.predict(img_array)
#     predicted_index = np.argmax(preds[0])

#     test_dir = os.path.join(r"D:\PYTHON\CropsDiseaseProject\Data_Split\test", crop)
#     class_labels = sorted(os.listdir(test_dir))
#     predicted_label = class_labels[predicted_index]
#     disease_name = predicted_label.replace(f"{crop}_", "")

#     return {
#         "crop": crop,
#         "disease": disease_name,
#         "confidence": f"{np.max(preds[0]) * 100:.2f}%",
#         "solution": "Apply proper pesticide/fungicide as per agricultural guidelines."
#     }

# # ======================================================
# # /predict Endpoint — Enhanced with Leaf Validation
# # ======================================================
# @app.post("/predict")
# async def predict(crop: str = Form(...), image: UploadFile = File(...)):
#     """
#     Main endpoint:
#     1️⃣ Verify leaf vs non-leaf
#     2️⃣ Verify crop type (optional)
#     3️⃣ Predict disease
#     """
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")

#         # Step 1️⃣ — Verify leaf vs non-leaf
#         is_leaf = verify_leaf_image(img)
#         if not is_leaf:
#             return {
#                 "status": "error",
#                 "message": "Invalid image — please upload a clear leaf image."
#             }

#         # Step 2️⃣ — Verify crop type
#         is_valid_crop = verify_crop_type(img, crop)
#         if not is_valid_crop:
#             return {
#                 "status": "error",
#                 "message": f"Invalid image — does not belong to selected crop ({crop})."
#             }

#         # Step 3️⃣ — Predict disease
#         result = predict_crop_disease(crop, img)

#         # Step 4️⃣ — Return structured JSON
#         return {"status": "success", "data": result}

#     except Exception as e:
#         return {"status": "error", "message": f"Server error: {str(e)}"}


# # ======================================================
# # 🧪 /test Endpoint (Optional)
# # ======================================================
# @app.post("/test")
# async def test_endpoint(crop: str = Form(...), image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")

#         # Leaf verification
#         if not verify_leaf_image(img):
#             return {"status": "error", "message": "Invalid image — please upload a clear leaf image."}

#         # Crop verification
#         if not verify_crop_type(img, crop):
#             return {"status": "error", "message": "Invalid image — does not belong to selected crop."}

#         result = predict_crop_disease(crop, img)
#         return {"status": "success", "data": result}

#     except Exception as e:
#         return {"status": "error", "message": str(e)}


# # ======================================================
# # 🧩 Debug Endpoint — Check Model Output for Leaf Gate
# # ======================================================
# @app.post("/debug_leaf_check")
# async def debug_leaf_check(image: UploadFile = File(...)):
#     """Debug endpoint: check what your leaf/non-leaf model outputs."""
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")

#         img = img.resize((224, 224))
#         arr = keras_image.img_to_array(img)
#         arr = np.expand_dims(arr, axis=0) / 255.0
#         score = float(leaf_classifier.predict(arr)[0][0])

#         return {
#             "raw_output": score,
#             "meaning": (
#                 "If leaf images give LOW (~0.1) and non-leaf give HIGH (~0.9), "
#                 "set invert_output=True.\n"
#                 "If opposite, set invert_output=False."
#             )
#         }
#     except Exception as e:
#         return {"error": str(e)}






# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image as keras_image
# import os


# # Initialize FastAPI app

# app = FastAPI(title="Crop Disease Prediction API")

# # Enable CORS (for mobile app or React Native)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # Later restrict to app domain or IP
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Load all .h5 models

# MODEL_DIR = r"D:\PYTHON\CropsDiseaseProject\models"

# CROP_MODELS = {
#     "Chili": os.path.join(MODEL_DIR, "Chili_Model_FT.h5"),
#     "Maize": os.path.join(MODEL_DIR, "Maize_Model_FT.h5"),
#     "Pea": os.path.join(MODEL_DIR, "Pea_Model_FT.h5"),
#     "Potato": os.path.join(MODEL_DIR, "Potato_Model_FT.h5"),
#     "Rice": os.path.join(MODEL_DIR, "Rice_Model_FT.h5"),
#     "Sugarcane": os.path.join(MODEL_DIR, "Sugarcane_Model_FT.h5"),
#     "Tomato": os.path.join(MODEL_DIR, "Tomato_Model_FT.h5"),
#     "Wheat": os.path.join(MODEL_DIR, "Wheat_Model_FT.h5"),
# }

# models = {}
# for crop, model_path in CROP_MODELS.items():
#     if os.path.exists(model_path):
#         models[crop] = tf.keras.models.load_model(model_path)
#         print(f"✅ Loaded model for {crop}")
#     else:
#         print(f"⚠️ Model not found for {crop}: {model_path}")


# # Prediction Function

# def predict_crop_disease(crop: str, img: Image.Image):
#     if crop not in models:
#         return {"error": f"No model found for crop: {crop}"}

#     model = models[crop]

#     # Preprocess the image
#     img = img.resize((224, 224))
#     img_array = keras_image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     # Predict
#     preds = model.predict(img_array)
#     predicted_index = np.argmax(preds[0])

#     # Get label names (from test folders)
#     test_dir = os.path.join(r"D:\PYTHON\CropsDiseaseProject\Data_Split\test", crop)
#     class_labels = sorted(os.listdir(test_dir))
#     predicted_label = class_labels[predicted_index]

#     # Remove duplicated crop name prefix like "Pea_Pea_Healthy"
#     disease_name = predicted_label.replace(f"{crop}_", "")

#     return {
#         "crop": crop,
#         "disease": disease_name,
#         "confidence": f"{np.max(preds[0]) * 100:.2f}%",
#         "solution": "Apply proper pesticide/fungicide as per agricultural guidelines."
#     }

# # Test Endpoint

# @app.post("/test")
# async def test_endpoint(crop: str = Form(...), image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#         result = predict_crop_disease(crop, img)
#         return {"status": "success", "data": result}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# # Main Prediction Endpoint

# @app.post("/predict")
# async def predict(crop: str = Form(...), image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#         result = predict_crop_disease(crop, img)
#         return {"status": "success", "data": result}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}



import os
import io
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from google.cloud import vision


#  GOOGLE VISION API SETUP

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\PYTHON\CropsDiseaseProject\google_key\cropdiseaseproject-ba9d8d750f35.json"


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


# /predict Endpoint — with Google Vision validation

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


# /debug_leaf_labels Endpoint — test Vision API labels

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
