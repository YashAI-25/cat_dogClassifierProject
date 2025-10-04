from fastapi import FastAPI,UploadFile,File
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle


app=FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # yaha "*" matlab sab domains allowed hain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post('/predictimg')
async def predict_image(image_path:UploadFile = File(...)):
    model=pickle.load(open('DLimgmodel.pkl','rb'))
    # img = cv2.imread(image_path)
    file_bytes = await image_path.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image {image_path}")
        return

    img = cv2.resize(img,(100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    prediction = model.predict(img_array)[0][0]
    print(prediction)
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return{"lb":label,
           'conf':round(float(confidence*100)),
           'img': image_path
           }