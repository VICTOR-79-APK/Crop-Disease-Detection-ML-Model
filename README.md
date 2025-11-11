# Crop-Disease-Detection-ML-Model
A crop disease detection model based on ML and Deep Learning Algorithms. It scans the leaves of the crop and identify the diseases that are present.
---

🌿 Crop Disease Detection Model

> An AI-powered system for identifying crop diseases using Deep Learning and Computer Vision.
  
---

📖 Project Overview

The Crop Disease Detection System is a deep learning–based solution designed to help farmers and researchers identify crop diseases through image classification.
Using convolutional neural networks (CNNs) and transfer learning (MobileNetV2 / ResNet50), this system classifies plant leaf images into healthy or diseased categories with high accuracy.

The project includes:

🧠 AI model training for 8 crops

🌱 Leaf validation and noise cleaning

⚙ FastAPI backend for prediction

📱 React Native mobile app for end-users

---

🧩 Modules Overview

No.	Module	Description

1️⃣	Dataset Management	Collect and organize crop images for all eight crops.
2️⃣	Data Cleaning & Noise Removal	Remove noisy, duplicate, or low-quality images.
3️⃣	Data Augmentation	Enhance dataset diversity using transformations.
4️⃣	Data Splitting	Split data into Train/Validation/Test sets (80–10–10).
5️⃣	Model Training, Train MobileNetV2 models for each crop disease type.
6️⃣	Model Evaluation	Evaluate performance with accuracy, F1-score, and confusion matrix.
7️⃣	Leaf Validation Model	Validate if the uploaded image is a crop leaf or invalid object.
8️⃣	FastAPI Backend Server	Host all AI models with REST API endpoints for predictions.
9️⃣	React Native Frontend App	Mobile app for image upload, detection, and solution display.
🔟	Deployment Module	Deploy on cloud (Render).

---

⚙ Tech Stack

Category	Technologies

Languages	Python, JavaScript
Frameworks	TensorFlow, FastAPI, React Native
Libraries	NumPy, OpenCV, Pillow, Pandas, Matplotlib, Seaborn
Model Architectures	MobileNetV2, ResNet50
Tools	Jupyter Notebook, VS Code
Deployment	Render / Railway / AWS EC2
Version Control	Git & GitHub

---

* AI Models Used

Crop	Model	Format

Wheat	MobileNetV2	.h5 / .tflite
Rice	MobileNetV2	.h5 / .tflite
Maize	MobileNetV2	.h5 / .tflite
Potato	MobileNetV2	.h5 / .tflite
Tomato	MobileNetV2	.h5 / .tflite
Chili	MobileNetV2	.h5 / .tflite
Sugarcane	MobileNetV2	.h5 / .tflite
Pea	MobileNetV2	.h5 / .tflite

Additional:

leaf_nonleaf_classifier.h5 → Leaf vs Non-Leaf validation

crop_type_classifier.h5 → Crop identification (optional)

---

🧾 Dataset Structure

CropsDiseaseProject/
│
├── Data_Split/
│   ├── train/
│   │   ├── Wheat/
│   │   ├── Rice/
│   │   └── ...
│   ├── validation/
│   └── test/
│
├── models/
│   ├── Wheat_Model_FT.h5
│   ├── leaf_nonleaf_classifier.h5
│   └── ...
│
└── server.py

---

🚀 How to Run the Project

🧱 Backend (FastAPI Server)

1. Clone the repository:

git clone https://github.com/<your-username>/Crop-Disease-Detection.git
cd Crop-Disease-Detection

2. Create a virtual environment and install dependencies:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

3. Run the server:

uvicorn server:app --reload --host 0.0.0.0 --port 8000


4. Access API Docs:

http://localhost:8000/docs

---

📱 Frontend (React Native Mobile App)

1. Go to the frontend folder:

cd mobile_app

2. Start the app (Expo):

npm install
npm start

3. Update your API endpoint in PredictScreen.js:

const response = await fetch("http://<your-IP>:8000/predict", {...});

4. Test it by uploading a leaf image!

---

🧪 Sample Output

Input Image	Model Output

🌾 Wheat Leaf	Disease: Leaf Blight → Apply fungicide
🍅 Tomato Leaf	Disease: Bacterial Spot → Use copper-based spray
🧾 Invalid Object	⚠ “Invalid image — not a crop leaf.”

---

📊 Performance Metrics

Crop	Accuracy	F1 Score

Wheat	97.3%	0.96
Rice	96.8%	0.95
Maize	98.1%	0.97
Tomato	97.9%	0.96
Average	97.5%	0.96

---

☁ Deployment

You can deploy the FastAPI server on:

Render

Railway

AWS EC2 / Azure VM

Then connect your mobile app using the public API URL.

---

📚 Future Enhancements

🌾 Add more crop categories (e.g., Soybean, Cotton).

🤖 Integrate on-device TensorFlow Lite for offline predictions.

☁ Cloud storage for model updates.

🔔 Push notifications for disease alerts.

📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute it.

---

🌟 Acknowledgements

Special thanks to:

TensorFlow & Keras for model development

FastAPI for backend framework

Expo & React Native for mobile integration

Dataset sources from PlantVillage & Kaggle
