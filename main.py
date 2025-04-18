# FastAPI for creating the web application.
# File and UploadFile for handling file uploads.
# CORSMiddleware for handling Cross-Origin Resource Sharing (CORS).
# uvicorn for running the ASGI server.
# numpy for numerical operations.
# BytesIO and Image from PIL for image processing.
# tensorflow for machine learning model operations.
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "newPlant.h5"
MODEL_URL = "https://huggingface.co/Dev1ng/cd/resolve/main/newPlant.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['AppleApple_scab', 'AppleBlack_rot', 'AppleCedar_apple_rust', 'Applehealthy', 'Blueberryhealthy',
    'Cherry_(including_sour)Powdery_mildew', 'Cherry_(including_sour)healthy', 'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)Common_rust', 'Corn(maize)Northern_Leaf_Blight', 'Corn_(maize)healthy', 'GrapeBlack_rot',
    'GrapeEsca_(Black_Measles)', 'GrapeLeaf_blight_(Isariopsis_Leaf_Spot)', 'Grapehealthy', 'OrangeHaunglongbing_(Citrus_greening)',
    'PeachBacterial_spot', 'Peachhealthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,bellhealthy', 'PotatoEarly_blight',
    'PotatoLate_blight', 'Potatohealthy', 'Raspberryhealthy', 'Soybeanhealthy', 'SquashPowdery_mildew',
    'StrawberryLeaf_scorch', 'Strawberryhealthy', 'TomatoBacterial_spot', 'TomatoEarly_blight', 'TomatoLate_blight',
    'TomatoLeaf_Mold', 'TomatoSeptoria_leaf_spot', 'TomatoSpider_mites Two-spotted_spider_mite', 'TomatoTarget_Spot',
    'TomatoTomato_Yellow_Leaf_Curl_Virus', 'TomatoTomato_mosaic_virus', 'Tomatohealthy']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(BytesIO(await file.read())))
        image = tf.image.resize(image, [256, 256])
        img_batch = np.expand_dims(image, 0) / 255.0
        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        print("Prediction error:", e)
        return {
            "class": "Error",
            "confidence": 0.0
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
